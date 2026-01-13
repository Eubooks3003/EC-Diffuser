import numpy as np
import torch
import h5py

# ---------------------------
# Low-level math helpers
# ---------------------------

def _squeeze_hw(d):
    d = np.asarray(d)
    if d.ndim == 3:
        d = np.squeeze(d)
    if d.ndim != 2:
        d = d.reshape(d.shape[-2], d.shape[-1])
    return d.astype(np.float32)

def _looks_like_rigid(T: np.ndarray, tol=1e-2) -> bool:
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        return False
    if not np.allclose(T[3], [0, 0, 0, 1], atol=tol):
        return False
    R = T[:3, :3]
    cn = np.linalg.norm(R, axis=0)
    if np.any(cn > 3.0) or np.any(cn < 0.2):
        return False
    return True

def _recover_cam2world(T_cam_world_like: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Same logic as your offline script.
    - If rigid cam->world, keep it.
    - Else treat top 3x4 as P = K [R|t] (world->cam projection-ish), recover E=inv(K)P, invert.
    """
    T = np.asarray(T_cam_world_like, dtype=np.float64).reshape(4, 4)
    if _looks_like_rigid(T):
        return T.astype(np.float32)

    P = T[:3, :4]
    E = np.linalg.inv(K) @ P
    Tw2c = np.eye(4, dtype=np.float64)
    Tw2c[:3, :4] = E
    Tc2w = np.linalg.inv(Tw2c)
    return Tc2w.astype(np.float32)

def _depth_to_meters(depth_raw, near, far):
    d = _squeeze_hw(depth_raw)

    # int buffers -> normalize to [0,1]
    if np.issubdtype(depth_raw.dtype, np.integer):
        denom = 65535.0 if depth_raw.dtype == np.uint16 else 255.0
        d = d / denom

    dmax = float(np.nanmax(d)) if np.isfinite(d).any() else 0.0
    if dmax <= 1.01 and (near is not None) and (far is not None):
        d = near + d * (far - near)

    return d.astype(np.float32)

def _apply_T(T, pts):
    # T: (4,4), pts: (N,3)
    ones = np.ones((pts.shape[0], 1), np.float32)
    pts_h = np.concatenate([pts, ones], axis=1)
    out = (T.astype(np.float32) @ pts_h.T).T
    return out[:, :3]

def _crop_xyzrgb(xyz, rgb, bounds_xyz):
    # bounds_xyz = ((xmin,xmax),(ymin,ymax),(zmin,zmax))
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds_xyz
    m = (
        (xyz[:, 0] >= xmin) & (xyz[:, 0] <= xmax) &
        (xyz[:, 1] >= ymin) & (xyz[:, 1] <= ymax) &
        (xyz[:, 2] >= zmin) & (xyz[:, 2] <= zmax)
    )
    if not np.any(m):
        return xyz, rgb
    return xyz[m], rgb[m]

# ---------------------------
# Voxelization (fixed bounds)
# ---------------------------

def voxelize_xyzrgb_fixed_bounds(
    xyz, rgb, grid_dhw, bounds_xyz, with_occ=False, eps=1e-6
):
    """
    Fixed-bounds voxelization (this is what you want online).
    - xyz is already in world coordinates.
    - bounds_xyz is constant for training/inference consistency.
    """
    (D, H, W) = grid_dhw
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.float32)

    if xyz.shape[0] == 0:
        C = 4 if with_occ else 3
        return np.zeros((C, D, H, W), dtype=np.float32)

    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]
    rgb = rgb[finite] if rgb.shape[0] == finite.shape[0] else rgb
    if xyz.shape[0] == 0:
        C = 4 if with_occ else 3
        return np.zeros((C, D, H, W), dtype=np.float32)

    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds_xyz
    pmin = np.array([xmin, ymin, zmin], dtype=np.float32)
    pmax = np.array([xmax, ymax, zmax], dtype=np.float32)
    span = np.maximum(pmax - pmin, eps)

    p01 = (xyz - pmin[None, :]) / span[None, :]
    p01 = np.clip(p01, 0.0, 1.0)

    ix = np.clip((p01[:, 0] * (W - 1)).astype(np.int32), 0, W - 1)
    iy = np.clip((p01[:, 1] * (H - 1)).astype(np.int32), 0, H - 1)
    iz = np.clip((p01[:, 2] * (D - 1)).astype(np.int32), 0, D - 1)

    size = D * H * W
    lin = (iz * H + iy) * W + ix

    cnt = np.zeros((size,), dtype=np.int32)
    rs  = np.zeros((size,), dtype=np.float32)
    gs  = np.zeros((size,), dtype=np.float32)
    bs  = np.zeros((size,), dtype=np.float32)

    np.add.at(cnt, lin, 1)
    np.add.at(rs,  lin, rgb[:, 0])
    np.add.at(gs,  lin, rgb[:, 1])
    np.add.at(bs,  lin, rgb[:, 2])

    mask = cnt > 0
    rs[mask] /= cnt[mask]
    gs[mask] /= cnt[mask]
    bs[mask] /= cnt[mask]

    rgb_vol = np.stack([rs, gs, bs], axis=0).reshape(3, D, H, W)

    if with_occ:
        occ = mask.astype(np.float32).reshape(1, D, H, W)
        return np.concatenate([rgb_vol, occ], axis=0).astype(np.float32)
    return rgb_vol.astype(np.float32)

# ---------------------------
# Token packing (K=24)
# ---------------------------

def pack_tokens_k24(out, scale_max=10.0, obj_on_thresh=0.0):
    """
    Token format: [pos(3), scale(3), feat(F), obj_on(1)] => Dtok = 7 + F
    Expects out['z'] : (B,1,24,3)
            out['z_scale'] or out['mu_scale'] : (B,1,24,3) in log-space typically
            out['z_features'] or out['mu_features'] : (B,1,24,F)
            out['obj_on'] or out['mu_obj_on'] : (B,1,24,1)
    """
    z = out.get("z", None)
    if z is None or z.dim() != 4 or z.shape[2] != 24 or z.shape[-1] != 3:
        raise RuntimeError(f"Expected out['z'] shape (B,1,24,3), got {None if z is None else tuple(z.shape)}")
    pos = z[:, 0]  # (B,24,3)

    z_scale = out.get("z_scale", None)
    if z_scale is None:
        z_scale = out.get("mu_scale", None)
    if z_scale is None or z_scale.dim() != 4 or z_scale.shape[2] != 24:
        scale = torch.zeros_like(pos)
    else:
        scale_raw = z_scale[:, 0]           # (B,24,3)
        scale = torch.exp(scale_raw)        # log->positive
        scale = scale.clamp(min=1e-6, max=float(scale_max))

    feat = out.get("z_features", None)
    if feat is None:
        feat = out.get("mu_features", None)
    if feat is None:
        feat = torch.zeros((pos.shape[0], 24, 0), device=pos.device, dtype=pos.dtype)
    else:
        if feat.dim() != 4 or feat.shape[2] != 24:
            raise RuntimeError(f"Expected features shape (B,1,24,F), got {tuple(feat.shape)}")
        feat = feat[:, 0]  # (B,24,F)

    obj_on = out.get("obj_on", None)
    if obj_on is None:
        obj_on = out.get("mu_obj_on", None)
    if obj_on is None:
        obj_on = torch.ones((pos.shape[0], 24, 1), device=pos.device, dtype=pos.dtype)
    else:
        if obj_on.dim() != 4 or obj_on.shape[2] != 24:
            raise RuntimeError(f"Expected obj_on shape (B,1,24,1), got {tuple(obj_on.shape)}")
        obj_on = obj_on[:, 0][..., :1]  # (B,24,1)

    toks = torch.cat([pos, scale, feat, obj_on], dim=-1)  # (B,24,7+F)
    if obj_on_thresh > 0:
        toks = toks * (obj_on > obj_on_thresh).float()
    return toks

# ---------------------------
# Main wrapper
# ---------------------------

class MimicGenDLPWrapper:
    """
    Live env wrapper:
      raw obs (rgb/depth) -> fused point cloud -> fixed-bounds voxel -> 3D DLP -> tokens

    It returns *flattened* obs vectors shaped [K*Dtok], suitable for EC-Diffuser conditioning.
    """

    def __init__(
        self,
        env,
        dlp_model,
        device,
        cams=("agentview", "sideview", "robot0_eye_in_hand"),
        grid_dhw=(64, 64, 64),
        bounds_xyz=((-2, 2), (-2, 2), (-0.2, 2.5)),
        with_occ=False,
        pixel_stride=2,
        max_points=200_000,
        calib_h5_path=None,
        obj_on_thresh=0.0,
        scale_max=10.0,
        get_goal_raw_obs_fn=None,
        success_key_candidates=("success", "task_success", "is_success"),
    ):
        self.env = env
        self.dlp = dlp_model
        self.device = torch.device(device)
        self.cams = list(cams)
        self.grid_dhw = tuple(grid_dhw)
        self.bounds_xyz = bounds_xyz
        self.with_occ = bool(with_occ)
        self.pixel_stride = int(pixel_stride)
        self.max_points = int(max_points) if max_points is not None else -1
        self.obj_on_thresh = float(obj_on_thresh)
        self.scale_max = float(scale_max)
        self.get_goal_raw_obs_fn = get_goal_raw_obs_fn
        self.success_key_candidates = tuple(success_key_candidates)

        # calib per cam: dict cam -> dict(K, Tc2w, near, far)
        self.calib = {}
        if calib_h5_path is not None:
            self._load_calib_from_h5(calib_h5_path)

        # precompute per-cam pixel grids for backprojection speed
        self._precomputed = {}  # cam -> (U, V, invfx, invfy, cx, cy)
        self.goal_vec = None  # flattened goal condition

    # ---------- calibration ----------
    def _load_calib_from_h5(self, h5_path):
        with h5py.File(h5_path, "r") as h5:
            for cam in self.cams:
                g = h5.get(f"meta/cameras/{cam}", None)
                if g is None:
                    raise KeyError(f"Missing meta/cameras/{cam} in {h5_path}")
                K = np.asarray(g["K"], dtype=np.float32)
                T_raw = np.asarray(g["T_cam_world"], dtype=np.float32)
                near = g.attrs.get("near", None)
                far  = g.attrs.get("far", None)
                near = float(near) if near is not None else None
                far  = float(far)  if far  is not None else None

                Tc2w = _recover_cam2world(T_raw, K.astype(np.float64))
                self.calib[cam] = {"K": K, "Tc2w": Tc2w, "near": near, "far": far}

    # ---------- obs extraction ----------
    def _get_rgbd_from_obs(self, obs, cam):
        """
        Tries common MimicGen / robomimic-style keys:
          {cam}_image, {cam}_depth
        """
        rgb_key = f"{cam}_image"
        dep_key = f"{cam}_depth"
        if rgb_key not in obs or dep_key not in obs:
            return None, None
        rgb = np.asarray(obs[rgb_key])
        depth = np.asarray(obs[dep_key])
        return rgb, depth

    # ---------- backprojection ----------
    def _ensure_precompute(self, cam, H, W):
        key = (cam, H, W)
        if key in self._precomputed:
            return self._precomputed[key]

        if cam not in self.calib:
            raise RuntimeError(
                f"No calibration for cam='{cam}'. Provide calib_h5_path or set self.calib[cam]."
            )
        K = self.calib[cam]["K"]
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])

        vv = np.arange(0, H, self.pixel_stride, dtype=np.int32)
        uu = np.arange(0, W, self.pixel_stride, dtype=np.int32)
        U, V = np.meshgrid(uu, vv)  # (H',W')

        pre = (U, V, 1.0 / fx, 1.0 / fy, cx, cy)
        self._precomputed[key] = pre
        return pre

    def _backproject_cam(self, cam, depth_raw):
        depth_m = _depth_to_meters(depth_raw, self.calib[cam]["near"], self.calib[cam]["far"])
        H, W = depth_m.shape

        U, V, invfx, invfy, cx, cy = self._ensure_precompute(cam, H, W)
        Z = depth_m[V, U]
        valid = np.isfinite(Z) & (Z > 0)

        if not np.any(valid):
            return np.zeros((0, 3), np.float32), np.zeros((0, 2), np.int32)

        Uv = U[valid].astype(np.float32)
        Vv = V[valid].astype(np.float32)
        Zv = Z[valid].astype(np.float32)

        X = (Uv - cx) * invfx * Zv
        Y = (Vv - cy) * invfy * Zv
        pts_cam = np.stack([X, Y, Zv], axis=-1).astype(np.float32)

        idxs = np.stack([V[valid], U[valid]], axis=-1).astype(np.int32)
        return pts_cam, idxs

    # ---------- core encoding ----------
    def encode_tokens(self, raw_obs):
        all_xyz, all_rgb = [], []

        for cam in self.cams:
            rgb, depth = self._get_rgbd_from_obs(raw_obs, cam)
            if rgb is None or depth is None:
                continue

            pts_cam, idxs = self._backproject_cam(cam, depth)
            if pts_cam.shape[0] == 0:
                continue

            v = idxs[:, 0]
            u = idxs[:, 1]
            rgb_pts = rgb[v, u, :].astype(np.float32)
            if rgb_pts.max() > 1.5:
                rgb_pts /= 255.0

            pts_world = _apply_T(self.calib[cam]["Tc2w"], pts_cam)

            all_xyz.append(pts_world)
            all_rgb.append(rgb_pts)

        if not all_xyz:
            raise RuntimeError("No points reconstructed from any camera. Check obs keys and calib.")

        xyz = np.concatenate(all_xyz, axis=0)
        rgb = np.concatenate(all_rgb, axis=0)

        # fixed world crop (keeps mapping stable)
        xyz, rgb = _crop_xyzrgb(xyz, rgb, self.bounds_xyz)

        # optional cap for speed
        if self.max_points > 0 and xyz.shape[0] > self.max_points:
            sel = np.random.default_rng(0).choice(xyz.shape[0], size=self.max_points, replace=False)
            xyz, rgb = xyz[sel], rgb[sel]

        vol = voxelize_xyzrgb_fixed_bounds(
            xyz, rgb, self.grid_dhw, self.bounds_xyz, with_occ=self.with_occ
        )  # (C,D,H,W)

        vox = torch.from_numpy(vol[None]).to(self.device)  # (1,C,D,H,W)
        with torch.no_grad():
            out = self.dlp(vox, deterministic=True, warmup=False, with_loss=False)
            toks = pack_tokens_k24(out, scale_max=self.scale_max, obj_on_thresh=self.obj_on_thresh)  # (1,24,Dtok)

        toks_np = toks[0].detach().cpu().numpy().astype(np.float32)  # (24,Dtok)
        flat = toks_np.reshape(-1)  # (24*Dtok,)
        return flat, toks_np, vol

    # ---------- goal handling ----------
    def _get_success_from_info(self, info):
        if info is None:
            return None
        for k in self.success_key_candidates:
            if k in info:
                v = info[k]
                if isinstance(v, (bool, np.bool_)):
                    return bool(v)
                if isinstance(v, (int, float, np.integer, np.floating)):
                    return bool(v)
        return None

    def reset(self, **kwargs):
        raw_obs = self.env.reset(**kwargs)

        obs_vec, _, _ = self.encode_tokens(raw_obs)

        # goal image/state -> encode once
        if self.get_goal_raw_obs_fn is not None:
            goal_raw = self.get_goal_raw_obs_fn(self.env, raw_obs)
            goal_vec, _, _ = self.encode_tokens(goal_raw)
            self.goal_vec = goal_vec
        else:
            # fallback: goal = current obs (debug only)
            self.goal_vec = obs_vec.copy()

        return obs_vec

    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        obs_vec, _, _ = self.encode_tokens(raw_obs)
        success = self._get_success_from_info(info)
        if success is not None:
            info = dict(info)
            info["success"] = success
        return obs_vec, reward, done, info

    # ---------- condition dict for diffusion ----------
    def make_cond(self, obs_vec, horizon):
        """
        Returns EC-Diffuser-style conditioning dict:
          {0: obs0, H-1: goal}
        """
        if self.goal_vec is None:
            raise RuntimeError("Call reset() first so goal_vec is available.")
        return {
            0: obs_vec.copy(),
            horizon - 1: self.goal_vec.copy(),
        }
