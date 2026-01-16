import numpy as np
import torch
import h5py

from datasets.voxelize_ds_wrapper import VoxelGridXYZ

# ----------------------------
# PLY-equivalent preprocessing
# ----------------------------

def center_scale_unit_cube_np(pts: np.ndarray) -> np.ndarray:
    """
    Identical math to your preprocessing script (but numpy).
    pts: [N,3] or [N,6] (xyz[0:3], rgb[3:6] optional)
    """
    if pts.ndim != 2 or pts.shape[1] not in (3, 6):
        raise RuntimeError(f"pts must be [N,3] or [N,6], got {pts.shape}")
    out = pts.copy()
    xyz = out[:, :3]
    if xyz.shape[0] == 0:
        return out
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = (mins + maxs) / 2.0
    scale = (maxs - mins).max() + 1e-8
    out[:, :3] = (xyz - center[None, :]) / scale * 2.0
    return out

def downsample_np(pts: np.ndarray, max_points: int) -> np.ndarray:
    if max_points is None or max_points <= 0:
        return pts
    if pts.shape[0] <= max_points:
        return pts
    idx = np.random.choice(pts.shape[0], max_points, replace=False)
    return pts[idx]

def parse_fixed_bounds_pm(bounds_str: str):
    # expects "xmin,xmax,ymin,ymax,zmin,zmax"
    xmin, xmax, ymin, ymax, zmin, zmax = map(float, bounds_str.split(","))
    pmin = (xmin, ymin, zmin)
    pmax = (xmax, ymax, zmax)
    return (pmin, pmax)

# ----------------------------
# Token packing (exactly like preprocessing)
# ----------------------------

def pack_tokens_k24_preproc_format(out: dict) -> torch.Tensor:
    """
    EXACT preprocessing format:
      toks = [z(3), z_scale(3), z_depth(1), obj_on(1), z_features(F)]
    Shapes expected: (B,1,24,*)
    """
    if "z" not in out:
        raise RuntimeError("DLP output missing key 'z'")


    z       = out["z"][:, 0]          # (B,24,3)
    z_scale = out["z_scale"][:, 0]    # (B,24,3)
    z_depth = out["z_depth"][:, 0]    # (B,24,3)
    obj_on  = out["obj_on"][:, 0]     # (B,24,1) or (B,24)
    z_feat  = out["z_features"][:, 0] # (B,24,F)

    if obj_on.ndim == 2:
        obj_on = obj_on.unsqueeze(-1)

    toks = torch.cat([z, z_scale, z_depth, obj_on, z_feat], dim=-1)  # (B,24,Dtok)
    return toks

# ----------------------------
# Depth/RGB helpers (fail-fast)
# ----------------------------

def _squeeze_hw(d):
    d = np.asarray(d)
    if d.ndim == 3:
        d = np.squeeze(d)
    if d.ndim != 2:
        d = d.reshape(d.shape[-2], d.shape[-1])
    return d.astype(np.float32)

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
    ones = np.ones((pts.shape[0], 1), np.float32)
    pts_h = np.concatenate([pts, ones], axis=1)
    out = (T.astype(np.float32) @ pts_h.T).T
    return out[:, :3]

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
    T = np.asarray(T_cam_world_like, dtype=np.float64).reshape(4, 4)
    if _looks_like_rigid(T):
        return T.astype(np.float32)

    P = T[:3, :4]
    E = np.linalg.inv(K) @ P
    Tw2c = np.eye(4, dtype=np.float64)
    Tw2c[:3, :4] = E
    Tc2w = np.linalg.inv(Tw2c)
    return Tc2w.astype(np.float32)

# ----------------------------
# Wrapper-faithful MimicGenDLPWrapper
# ----------------------------

class MimicGenDLPWrapper:
    """
    Online wrapper that matches preprocessing:
      RGB-D -> xyzrgb (world) -> (optional) unit-cube norm -> downsample ->
      voxelize via VoxelGridXYZ -> DLP -> pack_tokens_k24_preproc_format -> flatten

    IMPORTANT:
      - For training/inference consistency you should use bounds_mode='global' or 'fixed',
        NOT 'per_item'. But we expose the same options as preprocessing.
    """

    def __init__(
        self,
        env,
        dlp_model,
        device,
        cams=("agentview", "sideview", "robot0_eye_in_hand"),

        # preprocessing-equivalent knobs
        max_points=4096,
        normalize_to_unit_cube=False,   # MUST match what you used offline
        include_rgb=True,               # DLP expects avg_rgb voxel usually

        # voxelization knobs (same as preprocessing script)
        grid_dhw=(64, 64, 64),
        voxel_mode="avg_rgb",           # "avg_rgb"|"occupancy"|"density"|"moments"
        bounds_mode="per_item",           # "per_item"|"global"|"fixed"
        fixed_bounds="-1,1,-1,1,-1,1",  # used if bounds_mode="fixed"
        global_bounds_pm=None,          # required if bounds_mode="global"

        # calibration
        calib_h5_path=None,

        # goal
        get_goal_raw_obs_fn=None,
        success_key_candidates=("success", "task_success", "is_success"),
        pixel_stride=2,
        max_points_backproject=200_000,
    ):
        self.env = env
        self.dlp = dlp_model
        self.device = torch.device(device)
        self.cams = list(cams)

        self.max_points = int(max_points)
        self.normalize_to_unit_cube = bool(normalize_to_unit_cube)
        self.include_rgb = bool(include_rgb)

        self.grid_dhw = tuple(grid_dhw)
        self.voxel_mode = str(voxel_mode)
        self.bounds_mode = str(bounds_mode)
        self.pixel_stride = int(pixel_stride)
        self.max_points_backproject = int(max_points_backproject)

        self.get_goal_raw_obs_fn = get_goal_raw_obs_fn
        self.success_key_candidates = tuple(success_key_candidates)

        self.last_raw_obs = None
        self.last_info = None
        self.goal_vec = None

        # --- load calib ---
        self.calib = {}
        if calib_h5_path is None:
            raise RuntimeError("calib_h5_path is required for RGB-D backprojection online.")
        self._load_calib_from_h5(calib_h5_path)

        # --- bounds_pm consistent with preprocessing ---
        if self.bounds_mode == "fixed":
            self.bounds_pm = parse_fixed_bounds_pm(fixed_bounds)
        elif self.bounds_mode == "global":
            if global_bounds_pm is None:
                raise RuntimeError("bounds_mode='global' requires global_bounds_pm=(pmin_tuple,pmax_tuple) computed offline.")
            if not (isinstance(global_bounds_pm, (tuple, list)) and len(global_bounds_pm) == 2):
                raise RuntimeError(f"global_bounds_pm must be (pmin,pmax), got {type(global_bounds_pm)}")
            self.bounds_pm = (tuple(global_bounds_pm[0]), tuple(global_bounds_pm[1]))
        elif self.bounds_mode == "per_item":
            self.bounds_pm = None
        else:
            raise RuntimeError(f"Unknown bounds_mode={self.bounds_mode}")

        # precompute pixel grids cache
        self._precomputed = {}  # (cam,H,W) -> (U,V,invfx,invfy,cx,cy)

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
        rgb_key = f"{cam}_image"
        dep_key = f"{cam}_depth"
        if rgb_key not in obs:
            raise KeyError(f"obs missing key {rgb_key}")
        if dep_key not in obs:
            raise KeyError(f"obs missing key {dep_key}")
        return np.asarray(obs[rgb_key]), np.asarray(obs[dep_key])

    # ---------- backprojection ----------
    def _ensure_precompute(self, cam, H, W):
        key = (cam, H, W)
        if key in self._precomputed:
            return self._precomputed[key]
        K = self.calib[cam]["K"]
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])

        vv = np.arange(0, H, self.pixel_stride, dtype=np.int32)
        uu = np.arange(0, W, self.pixel_stride, dtype=np.int32)
        U, V = np.meshgrid(uu, vv)
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

    # ---------- preprocessing-faithful pipeline ----------
    def _fuse_xyzrgb_world(self, raw_obs) -> np.ndarray:
        all_xyz, all_rgb = [], []
        for cam in self.cams:
            rgb, depth = self._get_rgbd_from_obs(raw_obs, cam)
            pts_cam, idxs = self._backproject_cam(cam, depth)
            if pts_cam.shape[0] == 0:
                continue

            v = idxs[:, 0]
            u = idxs[:, 1]

            # normalize rgb to HWC float in [0,1]
            if rgb.ndim == 4 and rgb.shape[0] == 1:
                rgb = rgb[0]
            if rgb.ndim == 3 and rgb.shape[0] in (1, 3, 4) and rgb.shape[-1] not in (1, 3, 4):
                rgb = np.transpose(rgb, (1, 2, 0))
            if rgb.ndim != 3 or rgb.shape[2] < 3:
                raise RuntimeError(f"RGB for cam={cam} must be HWC with >=3 channels, got {rgb.shape}")

            rgb_pts = rgb[v, u, :3].astype(np.float32)
            if rgb_pts.max() > 1.5:
                rgb_pts /= 255.0

            pts_world = _apply_T(self.calib[cam]["Tc2w"], pts_cam)

            all_xyz.append(pts_world)
            all_rgb.append(rgb_pts)

        if len(all_xyz) == 0:
            raise RuntimeError("No points reconstructed from any camera (all empty).")

        xyz = np.concatenate(all_xyz, axis=0)
        rgb = np.concatenate(all_rgb, axis=0)

        if not np.isfinite(xyz).all():
            finite = np.isfinite(xyz).all(axis=1)
            xyz = xyz[finite]
            rgb = rgb[finite]
        if xyz.shape[0] == 0:
            raise RuntimeError("All reconstructed points were non-finite.")

        # optional speed cap BEFORE TODataset-like downsample
        if self.max_points_backproject > 0 and xyz.shape[0] > self.max_points_backproject:
            sel = np.random.choice(xyz.shape[0], self.max_points_backproject, replace=False)
            xyz, rgb = xyz[sel], rgb[sel]

        if self.include_rgb:
            return np.concatenate([xyz, rgb], axis=-1).astype(np.float32)  # [N,6]
        return xyz.astype(np.float32)  # [N,3]

    def _preprocess_points_like_TODataset(self, pts: np.ndarray) -> torch.Tensor:
        # pts: [N,3] or [N,6] float32
        if self.normalize_to_unit_cube:
            pts = center_scale_unit_cube_np(pts)
        pts = downsample_np(pts, self.max_points)
        return torch.from_numpy(pts).float()  # CPU

    def _voxelize_via_wrapper(self, pts_t: torch.Tensor) -> torch.Tensor:
        D, H, W = map(int, self.grid_dhw)

        if pts_t.ndim != 2 or pts_t.shape[1] not in (3, 6):
            raise RuntimeError(f"pts_t must be [N,3] or [N,6], got {tuple(pts_t.shape)}")

        # move to device (VoxelGridXYZ handles torch tensors)
        pts_t = pts_t.to(self.device)

        xyz = pts_t[:, :3]
        colors = None
        if self.voxel_mode == "avg_rgb":
            if pts_t.shape[1] != 6:
                raise RuntimeError("voxel_mode='avg_rgb' requires include_rgb=True and points [N,6].")
            colors = pts_t[:, 3:6]

        vg = VoxelGridXYZ(
            points_xyz=xyz,
            colors=colors,
            grid_whd=(W, H, D),     # wrapper expects (W,H,D)
            bounds=self.bounds_pm,  # None => per-item; else fixed/global
            mode=self.voxel_mode,
        )
        vox = vg.to_dense()  # [C,D,H,W]
        return vox

    @torch.no_grad()
    def encode_tokens(self, raw_obs):
        pts = self._fuse_xyzrgb_world(raw_obs)                 # [N,3] or [N,6] numpy
        pts_t = self._preprocess_points_like_TODataset(pts)    # CPU torch
        vox = self._voxelize_via_wrapper(pts_t)                # [C,D,H,W] device
        vox_b = vox.unsqueeze(0)                               # [1,C,D,H,W]

        out = self.dlp(vox_b, deterministic=True, warmup=False, with_loss=False)

        toks = pack_tokens_k24_preproc_format(out)             # [B,24,Dtok]

        toks_np = toks[0].detach().cpu().numpy().astype(np.float32)  # (24,Dtok)
        flat = toks_np.reshape(-1).astype(np.float32)
        vox_np = vox.detach().cpu().numpy().astype(np.float32)
        return flat, toks_np, vox_np

    # ---------- goal / env ----------
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
        self.last_raw_obs = raw_obs
        self.last_info = None

        obs_vec, _, _ = self.encode_tokens(raw_obs)

        if self.get_goal_raw_obs_fn is not None:
            goal_raw = self.get_goal_raw_obs_fn(self.env, raw_obs)
            goal_vec, _, _ = self.encode_tokens(goal_raw)
            self.goal_vec = goal_vec
        else:
            self.goal_vec = obs_vec.copy()

        return obs_vec

    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        self.last_raw_obs = raw_obs
        self.last_info = info

        obs_vec, _, _ = self.encode_tokens(raw_obs)
        success = self._get_success_from_info(info)
        if success is not None:
            info = dict(info)
            info["success"] = success
        return obs_vec, float(reward), bool(done), info

    def make_cond(self, obs_vec, horizon):
        if self.goal_vec is None:
            raise RuntimeError("Call reset() before make_cond().")
        return {
            0: obs_vec.copy(),
            horizon - 1: self.goal_vec.copy(),
        }
