"""
RLBench env wrapper for EC-Diffuser language-conditioned 2D-DLP rollouts.

Live PyRep/CoppeliaSim integration. Uses the raw `rlbench.environment.Environment`
API (no YARR dependency). Action convention exposed by RLBench tasks via
`MoveArmThenGripper(EndEffectorPoseViaPlanning, Discrete)` is

    [pos(3), quat(4, xyzw), gripper_open(1)]   (8D)

The diffusion policy emits actions in the 10D format used during preprocessing:

    [pos(3), rot6d(6), gripper_open(1)]

so this wrapper converts rot6d -> quat at step time. See `_action_to_rlbench`.

Online observation flow:
    task.reset() -> Observation
    -> grab 5 RGB views (front, overhead, left_shoulder, right_shoulder, wrist)
    -> stack into [V, H, W, 3] uint8
    -> caller-supplied `dlp_encode_fn(rgbs)` returns (tokens (K, Dtok), bg (bg_dim,))
    -> wrap into the dict the LanguageConditionedPolicy expects.

Why a callback for DLP encoding: the 2D DLP encoder lives in lpwm-dev/lpwm-copy
and pulling it into EC-Diffuser would create a hard dep. The training pipeline
already loads dlp_ckpt + dlp_cfg from disk; the rollout entrypoint passes a
small wrapper that runs that encoder per-frame and returns the 2D-DLP tokens.
"""
import re

import numpy as np


RLBENCH_TASKS = [
    "close_jar",
    "open_drawer",
    "sweep_to_dustpan_of_size",
    "meat_off_grill",
    "turn_tap",
    "slide_block_to_color_target",
    "put_item_in_drawer",
    "reach_and_drag",
    "push_buttons",
    "stack_blocks",
]

DEFAULT_CAMS = ["front", "overhead", "left_shoulder", "right_shoulder"]


def extract_rlbench_task_name(pkl_or_calib_path: str) -> str:
    path = pkl_or_calib_path.lower()
    for name in RLBENCH_TASKS:
        if name in path:
            return name
    m = re.search(r"rlbench[_/]([a-z_]+?)(?:_d\d+|/|\.|$)", path)
    if m:
        return m.group(1)
    raise RuntimeError(f"Cannot extract RLBench task name from: {pkl_or_calib_path}")


def _snake_to_pascal(name: str) -> str:
    return "".join(p.capitalize() for p in name.split("_"))


def _resolve_task_class(task_name: str):
    """Look up the RLBench task class for a snake_case task name."""
    import importlib
    module = importlib.import_module(f"rlbench.tasks.{task_name}")
    return getattr(module, _snake_to_pascal(task_name))


def rot6d_to_quat_xyzw(rot6d: np.ndarray) -> np.ndarray:
    """Inverse of envs/mimicgen_dlp_wrapper.quat_to_rot6d.

    rot6d layout (matches preprocessing): first two columns of R, flattened as
    [r00, r10, r20, r01, r11, r21].

    Recovers an orthonormal R via Gram-Schmidt, then converts to (x,y,z,w) quat.
    """
    rot6d = np.asarray(rot6d, dtype=np.float64).reshape(6)
    a1 = rot6d[:3]
    a2 = rot6d[3:]
    b1 = a1 / (np.linalg.norm(a1) + 1e-12)
    a2_proj = a2 - np.dot(b1, a2) * b1
    b2 = a2_proj / (np.linalg.norm(a2_proj) + 1e-12)
    b3 = np.cross(b1, b2)
    R = np.stack([b1, b2, b3], axis=1)  # 3x3

    # Standard matrix->quat (Shepperd). Returns (x,y,z,w).
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif (m00 > m11) and (m00 > m22):
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    q /= (np.linalg.norm(q) + 1e-12)
    return q.astype(np.float32)


def quat_xyzw_to_rot6d(quat: np.ndarray) -> np.ndarray:
    """Same convention as mimicgen_dlp_wrapper.quat_to_rot6d (kept here so the
    RLBench gripper-state extractor doesn't need to import the mimicgen file)."""
    x, y, z, w = quat[0], quat[1], quat[2], quat[3]
    r00 = 1 - 2 * (y * y + z * z)
    r10 = 2 * (x * y + w * z)
    r20 = 2 * (x * z - w * y)
    r01 = 2 * (x * y - w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r21 = 2 * (y * z + w * x)
    return np.array([r00, r10, r20, r01, r11, r21], dtype=np.float32)


def _gripper_state_from_obs(obs) -> np.ndarray:
    """Build the 10D gripper-state vector [pos(3), rot6d(6), open(1)] from a
    raw RLBench Observation. Mirrors the preprocessing pipeline so the trained
    policy sees the same format at rollout time it saw during training."""
    pose = np.asarray(obs.gripper_pose, dtype=np.float32)  # (7,) [x,y,z, qx,qy,qz,qw]
    pos = pose[:3]
    quat = pose[3:7]
    rot6d = quat_xyzw_to_rot6d(quat)
    if obs.gripper_open is not None:
        gopen = float(obs.gripper_open)
    elif obs.gripper_joint_positions is not None:
        # Panda: 0=closed, ~0.04=open. Normalize to [0, 1].
        gopen = float(np.clip(np.mean(obs.gripper_joint_positions) / 0.04, 0.0, 1.0))
    else:
        gopen = 1.0
    return np.concatenate([pos, rot6d, np.array([gopen], dtype=np.float32)], axis=0)


def _stack_camera_rgbs(obs, cams=DEFAULT_CAMS) -> np.ndarray:
    """Pull the named RGB views off a RLBench Observation, stack as [V,H,W,3] uint8."""
    rgbs = []
    for cam in cams:
        attr = f"{cam}_rgb"
        rgb = getattr(obs, attr, None)
        if rgb is None:
            raise RuntimeError(
                f"RLBench Observation missing {attr}. Configure ObservationConfig "
                f"to enable the {cam} camera before launching the env."
            )
        rgbs.append(np.asarray(rgb))
    return np.stack(rgbs, axis=0).astype(np.uint8)


class RLBenchDLPEnv:
    """
    Live RLBench rollout environment for the language-conditioned 2D-DLP policy.

    Construction:
        task_name:        snake_case RLBench task (e.g. "close_jar").
        dlp_encode_fn:    callable(rgbs: np.ndarray [V,H,W,3] uint8) -> dict
                          with keys 'tokens' (K, Dtok) and 'bg' (bg_dim,).
                          Provided by the rollout entrypoint, which loads the
                          frozen 2D DLP from dlp_ckpt + dlp_cfg.
        cams:             list of RLBench camera names (default 5).
        image_size:       int, square image size for all enabled cameras.
        headless:         run CoppeliaSim without GUI.
        episode_length:   max steps per episode.
    """

    def __init__(self, task_name: str, dlp_encode_fn,
                 cams=None, image_size: int = 128, headless: bool = True,
                 episode_length: int = 400, action_mode_kwargs=None):
        self.task_name = task_name
        self.dlp_encode_fn = dlp_encode_fn
        self.cams = list(cams) if cams is not None else list(DEFAULT_CAMS)
        self.image_size = int(image_size)
        self.headless = headless
        self.episode_length = int(episode_length)
        self.action_mode_kwargs = action_mode_kwargs or {}

        self._env = None
        self._task = None
        self._task_class = None
        self._current_lang = None
        self._current_variation = None
        self._step_count = 0
        self._recorded_frames = []          # list of np.uint8 (H,W,3) for video
        self._recorded_tokens_front = []    # list of (K_front, Dtok) for kp overlay
        self._record_enabled = True

    # ------------------------------------------------------------------ launch
    def _launch(self):
        from rlbench import ObservationConfig, CameraConfig
        from rlbench.action_modes.action_mode import MoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
        from rlbench.action_modes.gripper_action_modes import Discrete
        from rlbench.environment import Environment

        obs_config = ObservationConfig()
        obs_config.set_all(False)
        obs_config.gripper_pose = True
        obs_config.gripper_open = True
        obs_config.gripper_joint_positions = True
        obs_config.joint_positions = True

        cam_attrs = {
            "front": "front_camera",
            "overhead": "overhead_camera",
            "left_shoulder": "left_shoulder_camera",
            "right_shoulder": "right_shoulder_camera",
            "wrist": "wrist_camera",
        }
        for cam in self.cams:
            attr = cam_attrs.get(cam)
            if attr is None:
                raise ValueError(f"Unknown RLBench camera name: {cam}")
            cc = CameraConfig(
                image_size=(self.image_size, self.image_size),
                rgb=True, depth=False, point_cloud=False, mask=False,
            )
            setattr(obs_config, attr, cc)

        # EndEffectorPoseViaPlanning: direct IK for per-frame dense control (horizon=5, exe_steps=1).
        # EndEffectorPoseViaPlanning would RRT-plan every small delta and frequently fail.
        action_mode = MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaPlanning(**self.action_mode_kwargs),
            gripper_action_mode=Discrete(),
        )

        self._env = Environment(action_mode=action_mode,
                                obs_config=obs_config,
                                headless=self.headless)
        self._env.launch()
        self._task_class = _resolve_task_class(self.task_name)
        self._task = self._env.get_task(self._task_class)

    def shutdown(self):
        if self._env is not None:
            try:
                self._env.shutdown()
            finally:
                self._env = None
                self._task = None

    # ------------------------------------------------------------------ obs
    def _make_obs_dict(self, rlbench_obs):
        rgbs = _stack_camera_rgbs(rlbench_obs, cams=self.cams)
        enc = self.dlp_encode_fn(rgbs)
        tokens = np.asarray(enc["tokens"], dtype=np.float32)         # (K, Dtok)
        bg = np.asarray(enc["bg"], dtype=np.float32).reshape(-1)     # (bg_dim,)
        gripper_state = _gripper_state_from_obs(rlbench_obs)         # (10,)
        # Front view is the first block of tokens (cams are concatenated in order,
        # and self.cams[0] == 'front' by convention).
        K_per_view = tokens.shape[0] // max(len(self.cams), 1)
        tokens_front = tokens[:K_per_view] if self.cams and self.cams[0] == "front" else None
        self._last_tokens_front = tokens_front
        return {
            "obs": tokens,
            "bg_features": bg,
            "gripper_state": gripper_state,
            "language": self._current_lang,
            "variation_number": self._current_variation,
        }

    # ------------------------------------------------------------------ reset
    def _record(self, rlbench_obs, tokens_front=None):
        if not self._record_enabled:
            return
        frame = getattr(rlbench_obs, "front_rgb", None)
        if frame is not None:
            arr = np.asarray(frame)
            # Normalize to (H, W, 3) uint8
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
            if arr.dtype != np.uint8:
                if arr.max() <= 1.5:
                    arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
                else:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
            if not getattr(self, "_logged_frame_shape", False):
                print(f"[RLBenchDLPEnv] front_rgb stored as shape={arr.shape} dtype={arr.dtype}",
                      flush=True)
                self._logged_frame_shape = True
            self._recorded_frames.append(arr)
        if tokens_front is not None:
            self._recorded_tokens_front.append(np.asarray(tokens_front, dtype=np.float32))

    def pop_recorded_frames(self):
        """Return (frames, front_tokens) and clear the recording buffer.
        front_tokens is a list of (K_front, Dtok) arrays aligned with frames."""
        frames = self._recorded_frames
        front_toks = getattr(self, "_recorded_tokens_front", [])
        self._recorded_frames = []
        self._recorded_tokens_front = []
        return frames, front_toks

    def reset(self, variation: int = None):
        if self._env is None:
            self._launch()
        if variation is None:
            variation = np.random.randint(self._task.variation_count())
        variation = int(variation) % self._task.variation_count()
        self._task.set_variation(variation)
        self._current_variation = variation
        descriptions, rlbench_obs = self._task.reset()
        self._current_lang = descriptions[0] if len(descriptions) > 0 else ""
        self._step_count = 0
        self._recorded_frames = []
        self._record(rlbench_obs, tokens_front=getattr(self, "_last_tokens_front", None))
        return self._make_obs_dict(rlbench_obs)

    # ------------------------------------------------------------------ step
    # Workspace clamp: RLBench rejects poses at/below the table as
    # InvalidActionError even if they're just ~mm below. Clamp z to a small
    # safety margin above the table. Values matched to training data +
    # RLBench Panda table height.
    _Z_MIN = 0.760   # training min was 0.758; a tiny ε above to be safe

    def _action_to_rlbench(self, action: np.ndarray) -> np.ndarray:
        """Convert the policy's 10D [pos(3), rot6d(6), open(1)] action into the
        8D RLBench EndEffectorPoseViaPlanning + Discrete format
        [pos(3), quat_xyzw(4), gripper_open(1)]."""
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.shape[0] != 10:
            raise ValueError(
                f"Expected 10D action [pos(3)+rot6d(6)+open(1)], got shape {a.shape}"
            )
        pos = a[:3].copy()
        # Clamp z above table to avoid InvalidActionError on near-table predictions.
        pos[2] = max(float(pos[2]), self._Z_MIN)
        quat = rot6d_to_quat_xyzw(a[3:9])
        gopen = np.array([1.0 if a[9] >= 0.5 else 0.0], dtype=np.float32)
        return np.concatenate([pos, quat, gopen], axis=0)

    def step(self, action):
        from rlbench.backend.exceptions import InvalidActionError
        try:
            from pyrep.errors import IKError, ConfigurationPathError
            _path_errs = (IKError, ConfigurationPathError, InvalidActionError)
        except Exception:
            _path_errs = (InvalidActionError,)

        rlb_action = self._action_to_rlbench(action)
        info = {"variation_number": self._current_variation,
                "language": self._current_lang}
        # Retry-on-fail: a single bad prediction shouldn't kill the rollout.
        # Mirror lpwm-occ's _Mover: up to MAX_RETRIES attempts on planning errors
        # before we give up and terminate.
        MAX_RETRIES = 10
        rlbench_obs, reward, terminal = None, 0.0, False
        caught = []
        for attempt in range(MAX_RETRIES):
            try:
                rlbench_obs, reward, terminal = self._task.step(rlb_action)
                break
            except _path_errs as e:
                caught.append(type(e).__name__)
                # stale last obs is fine as a placeholder; the policy will replan
                # from _last_tokens_front next env.step() anyway.
                continue
        if rlbench_obs is None:
            info["error"] = caught[-1] if caught else "PlanningError"
            info["retry_history"] = caught
            return None, 0.0, True, info
        if caught:
            info["retried"] = caught  # surfaces to eval log for diagnostics

        self._step_count += 1
        if self._step_count >= self.episode_length:
            terminal = True
        self._record(rlbench_obs, tokens_front=getattr(self, "_last_tokens_front", None))
        return self._make_obs_dict(rlbench_obs), float(reward), bool(terminal), info

    # ------------------------------------------------------------------ props
    @property
    def language(self) -> str:
        return self._current_lang

    @property
    def variation_number(self) -> int:
        return self._current_variation
