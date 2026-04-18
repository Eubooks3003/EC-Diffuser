"""
RLBench env wrapper for EC-Diffuser language-conditioned rollouts.

Minimal interface contract expected by LanguageConditionedPolicy:
    env.reset() -> dict with keys:
        'obs':             DLP latent particle tokens (K, D) at t=0
        'gripper_state':   (10,) current EEF state [pos(3), rot6d(6), open(1)]
        'bg_features':     (bg_dim,) DLP background features
        'language':        str -- the instruction (from variation_descriptions.pkl,
                                                    or Task.get_base_task().init_episode)
        'variation_number':int

    env.step(action) -> (next_obs_dict, reward, done, info)
        action: np.ndarray shape (10,) = [pos(3), rot6d(6), open(1)], absolute EEF.

Actually wiring this up requires:
  - rlbench + CoppeliaSim (PyRep) installed and running
  - a live DLP encoder matching the preprocessing one
  - an online voxelization pipeline (fused point cloud -> voxel grid) matching
    lpwm-dev/scripts/preprocess_rlbench_voxels.py

Those live outside EC-Diffuser. This file provides the dataclass + task name
utilities the policy side depends on, and leaves the online encoding step as a
hook the user must fill in for their particular rollout harness.
"""
import re


# Canonical RLBench task names we have data / configs for.
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


def extract_rlbench_task_name(pkl_or_calib_path: str) -> str:
    """
    Infer RLBench task name from a dataset path. Mirrors the mimicgen helper
    `extract_mimicgen_task_name` so EC-Diffuser code that dispatches on task
    name (voxel bounds, camera config, etc.) has an analogous entry point.
    """
    path = pkl_or_calib_path.lower()
    for name in RLBENCH_TASKS:
        if name in path:
            return name
    # Fallback: match patterns like "rlbench_<task>_d0" or "rlbench/<task>/"
    m = re.search(r"rlbench[_/]([a-z_]+?)(?:_d\d+|/|\.|$)", path)
    if m:
        return m.group(1)
    raise RuntimeError(f"Cannot extract RLBench task name from: {pkl_or_calib_path}")


class RLBenchDLPEnv:
    """
    Skeleton language-conditioned RLBench rollout env. Fill in _encode_obs and
    _apply_action with a live PyRep/RLBench session for actual rollouts.

    Construction takes the same frozen DLP encoder used during preprocessing,
    plus camera / voxel configuration matching the preprocessing pipeline.
    """

    def __init__(self, task_name: str, dlp_model, dlp_cfg,
                 voxel_bounds=None, device: str = "cuda:0"):
        self.task_name = task_name
        self.dlp_model = dlp_model
        self.dlp_cfg = dlp_cfg
        self.voxel_bounds = voxel_bounds
        self.device = device

        self._rlbench_env = None   # populated by self._launch() in subclass / user code
        self._task = None
        self._current_lang = None
        self._current_variation = None

    def _launch(self):
        raise NotImplementedError(
            "User must subclass RLBenchDLPEnv and implement _launch() to start "
            "rlbench.environment.Environment + task setup with PyRep."
        )

    def _voxelize_current_scene(self):
        """Build the same sparse voxel grid the preprocessor produced, from the
        live RLBench cameras (fuse point clouds, center/scale to unit cube,
        voxelize). Must match preprocess_rlbench_voxels.py exactly."""
        raise NotImplementedError("Implement using lpwm-dev voxelization code.")

    def _encode_obs(self, vox):
        """Run the frozen DLP encoder on a single-frame voxel grid and return
        (tokens: (K, Dtok), bg: (bg_dim,), gripper_state: (10,))."""
        raise NotImplementedError("Implement using preprocess_rlbench pack_tokens_k24 logic.")

    def reset(self):
        if self._rlbench_env is None:
            self._launch()
        # RLBench pattern: env.get_demos(...) gives a list of Demos; for rollout
        # you typically reset_to_demo(demo) or task.sample_variation() + descriptions.
        # Pseudocode:
        #   descriptions, obs = self._task.reset()
        #   self._current_lang = descriptions[0]
        #   self._current_variation = self._task.variation_number
        raise NotImplementedError("reset() requires a live RLBench task handle.")

    def step(self, action):
        raise NotImplementedError("step() requires a live RLBench task handle.")

    @property
    def language(self) -> str:
        return self._current_lang

    @property
    def variation_number(self) -> int:
        return self._current_variation
