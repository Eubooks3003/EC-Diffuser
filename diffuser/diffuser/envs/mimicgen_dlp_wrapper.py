import numpy as np
import torch
import h5py
from robosuite.utils import camera_utils as CU

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


    z       = out["z"][:, 0]          
    z_scale = out["z_scale"][:, 0]    
    z_depth = out["z_depth"][:, 0]    
    obj_on  = out["obj_on"][:, 0]    
    z_feat  = out["z_features"][:, 0] 

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
        crop_bounds={"xmin":-0.5,"xmax":1.5,"ymin":-0.5,"ymax":1.5,"zmin":-0.2,"zmax":2.5},

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

        self.last_pts_world = None          # [N,3] float32 (optional)
        self.last_pts_preproc = None        # [M,3] or [M,6] float32 after unit-cube/downsample
        self.last_vox = None                # [C,D,H,W] float32 (np)
        self.last_toks = None               # [K,Dtok] float32 (np)
        self.last_obs_vec = None            # flat [K*Dtok] float32 (np)
        self.last_raw_obs = None
        self.last_info = None
        self.goal_vec = None

        # --- load calib ---
        self.calib = {}
        self.crop_bounds = crop_bounds 
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
    def _get_robosuite_env_and_sim(self):
        """
        Unwrap robomimic EnvRobosuite (and any other wrappers) to reach a robosuite env with .sim.
        Returns (robosuite_env, sim).
        """
        e = self.env
        seen = set()

        # unwrap a few common wrapper fields
        for _ in range(10):
            if id(e) in seen:
                break
            seen.add(id(e))

            if hasattr(e, "sim"):
                return e, e.sim

            # robomimic EnvRobosuite typically stores the underlying robosuite env here
            for attr in ("env", "_env", "base_env", "unwrapped"):
                if hasattr(e, attr):
                    e2 = getattr(e, attr)
                    # some wrappers have unwrapped as property returning itself; guard
                    if e2 is not None and e2 is not e:
                        e = e2
                        break
            else:
                # none of the attrs existed
                break

        raise AttributeError(
            f"Could not find underlying robosuite env with `.sim`. "
            f"Got wrapper chain ending at type={type(e)} with attrs={dir(e)}"
        )
    def _ensure_env_calib(self, raw_obs):
        robosuite_env, sim = self._get_robosuite_env_and_sim()

        # OpenCV -> OpenGL camera-frame conversion (x stays, y/z flip)
        F = np.eye(4, dtype=np.float32)
        F[1, 1] = -1.0
        F[2, 2] = -1.0

        for cam in self.cams:
            depth = np.asarray(raw_obs[f"{cam}_depth"])
            if depth.ndim == 3 and depth.shape[0] == 1:
                depth = depth[0]
            H, W = int(depth.shape[-2]), int(depth.shape[-1])

            prev = self.calib.get(cam, None)
            if prev is not None and prev["H"] == H and prev["W"] == W:
                continue

            # Intrinsics at this resolution
            K = CU.get_camera_intrinsic_matrix(
                sim, camera_name=cam, camera_height=H, camera_width=W
            ).astype(np.float32)

            # ---- MuJoCo camera pose (cam frame in world coords) ----
            # Prefer modern mujoco API if available
            if hasattr(sim.data, "get_camera_xpos") and hasattr(sim.data, "get_camera_xmat"):
                pos = sim.data.get_camera_xpos(cam).copy().astype(np.float32)            # (3,)
                R_c2w_gl = sim.data.get_camera_xmat(cam).copy().reshape(3, 3).astype(np.float32)
            else:
                # mujoco-py style arrays
                try:
                    cid = sim.model.camera_name2id(cam)
                    pos = sim.data.cam_xpos[cid].copy().astype(np.float32)
                    R_c2w_gl = sim.data.cam_xmat[cid].copy().reshape(3, 3).astype(np.float32)
                except Exception as e:
                    raise AttributeError(
                        f"Cannot access camera pose for cam='{cam}'. "
                        f"Tried sim.data.get_camera_xpos/xmat and sim.data.cam_xpos/xmat. Error: {e}"
                    )

            Tc2w_gl = np.eye(4, dtype=np.float32)
            Tc2w_gl[:3, :3] = R_c2w_gl
            Tc2w_gl[:3,  3] = pos

            # Convert from OpenCV cam coords (your backprojection) to MuJoCo/OpenGL cam coords
            # world = Tc2w_gl * (F * p_cv)
            Tc2w = Tc2w_gl @ F

            near = float(sim.model.vis.map.znear)
            far  = float(sim.model.vis.map.zfar)

            self.calib[cam] = {"K": K, "Tc2w": Tc2w, "near": near, "far": far, "H": H, "W": W}

    def _depth_to_meters_env(self, depth_raw):
        d = np.asarray(depth_raw)
        if d.ndim == 3 and d.shape[0] == 1:
            d = d[0]
        # robosuite expects a depth buffer in [0,1]
        _, sim = self._get_robosuite_env_and_sim()
        return CU.get_real_depth_map(sim, d).astype(np.float32)

    def crop_world(self, xyz, rgb=None):
        b = self.crop_bounds
        if b is None:
            return (xyz, rgb) if rgb is not None else xyz

        m = (
            (xyz[:, 0] >= b["xmin"]) & (xyz[:, 0] <= b["xmax"]) &
            (xyz[:, 1] >= b["ymin"]) & (xyz[:, 1] <= b["ymax"]) &
            (xyz[:, 2] >= b["zmin"]) & (xyz[:, 2] <= b["zmax"])
        )
        if not np.any(m):
            return (xyz, rgb) if rgb is not None else xyz
        xyz2 = xyz[m]
        if rgb is None:
            return xyz2
        return xyz2, rgb[m]


    # ---------- obs extraction ----------
    def _get_rgbd_from_obs(self, obs, cam):
        rgb_key = f"{cam}_image"
        dep_key = f"{cam}_depth"
        if rgb_key not in obs:
            raise KeyError(f"obs missing key {rgb_key}")
        if dep_key not in obs:
            raise KeyError(f"obs missing key {dep_key}")
        return np.asarray(obs[rgb_key]), np.asarray(obs[dep_key])

    def _ensure_precompute(self, cam, H, W):
        key = (cam, H, W, self.pixel_stride)
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
        depth_m = self._depth_to_meters_env(depth_raw)  # [H,W] meters
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

        xyz, rgb = self.crop_world(xyz, rgb)

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

    def _get_cam_calib_from_env(self, cam, raw_obs):
        sim = self.env.sim

        # infer current render size from the observation
        depth = np.asarray(raw_obs[f"{cam}_depth"])
        if depth.ndim == 3 and depth.shape[0] == 1:
            depth = depth[0]
        H, W = depth.shape[-2], depth.shape[-1]

        # intrinsics for *this* H,W
        K = CU.get_camera_intrinsic_matrix(sim, camera_name=cam, camera_height=H, camera_width=W).astype(np.float32)

        # extrinsics (check whether this returns world->cam or cam->world in YOUR robosuite version)
        # many robosuite versions provide this as world->camera
        Tw2c = CU.get_camera_extrinsic_matrix(sim, camera_name=cam).astype(np.float32)

        # convert to cam->world
        Tc2w = np.linalg.inv(Tw2c).astype(np.float32)

        # near/far for depth conversion
        # robosuite typically uses sim.model.vis.map.znear / zfar
        near = float(sim.model.vis.map.znear)
        far  = float(sim.model.vis.map.zfar)

        return K, Tc2w, near, far

    @torch.no_grad()
    def encode_tokens(self, raw_obs):
        # ---- 1) fused world points (pre-downsample) ----
        self._ensure_env_calib(raw_obs) 
        pts = self._fuse_xyzrgb_world(raw_obs)                 # [N,3] or [N,6] np
        # cache world xyz for debug (drop rgb if present)
        self.last_pts_world = pts[:, :3].astype(np.float32, copy=True)

        # ---- 2) TODataset-like preproc (unit cube + downsample) ----
        pts_t = self._preprocess_points_like_TODataset(pts)    # CPU torch [M,3/6]
        self.last_pts_preproc = pts_t.numpy().astype(np.float32, copy=True)

        # ---- 3) voxelize (THIS is the "GT voxel" fed to DLP) ----
        vox = self._voxelize_via_wrapper(pts_t)                # torch [C,D,H,W] on device
        self.last_vox = vox.detach().cpu().numpy().astype(np.float32, copy=True)

        # ---- 4) DLP encode -> tokens ----
        vox_b = vox.unsqueeze(0)                               # [1,C,D,H,W]
        out = self.dlp(vox_b, deterministic=True, warmup=False, with_loss=False)
        toks = pack_tokens_k24_preproc_format(out)             # [1,K,Dtok] torch
        toks_np = toks[0].detach().cpu().numpy().astype(np.float32, copy=True)  # [K,Dtok]
        self.last_toks = toks_np
        flat = toks_np.reshape(-1).astype(np.float32, copy=True)
        self.last_obs_vec = flat

        return flat, toks_np, self.last_vox

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
        self._ensure_env_calib(raw_obs)
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
