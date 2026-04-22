#!/usr/bin/env python
"""
Launch one frame via the ACTUAL RLBenchDLPEnv wrapper and via a stripped-down
Environment call, to see which wrapper kwarg (dataset_root? multi-cam?) flips
the floor from tan wood to washed-out yellow.

Usage:
    cd /home/ellina/Desktop/EC-Diffuser
    conda activate ecdiffuser
    xvfb-run -a python diffuser/scripts/compare_wrapper_vs_raw.py

Writes:
    /home/ellina/Desktop/render_compare/wrapper_front.png
    /home/ellina/Desktop/render_compare/raw_front.png
    /home/ellina/Desktop/render_compare/raw_multicam_front.png
    /home/ellina/Desktop/render_compare/raw_datasetroot_front.png
"""

import os
import sys
import numpy as np
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DIFFUSER_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
EC_DIFFUSER_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
for _p in [EC_DIFFUSER_ROOT, DIFFUSER_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT = "/home/ellina/Desktop/render_compare"
os.makedirs(OUT, exist_ok=True)

TASK = "close_jar"
IMG = 128
DATASET_ROOT = "/home/ellina/Desktop/data/rlbench_rgb"


def _save(arr, name):
    Image.fromarray(np.asarray(arr, dtype=np.uint8)).save(os.path.join(OUT, name))
    floor = np.asarray(arr)[64:, :, :].reshape(-1, 3).mean(axis=0).round(1)
    print(f"  {name}: floor_mean_rgb={floor}")


def raw(dataset_root=None, multi_cam=False,
        overhead_depth_only=False, fronts_only_overhead_removed=False,
        front_overhead=False, front_overhead_all_data=False,
        render_mode_name="OPENGL"):
    from rlbench import ObservationConfig, CameraConfig
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.environment import Environment
    from pyrep.const import RenderMode
    from diffuser.envs.rlbench_dlp_wrapper import _resolve_task_class

    rmode = {"OPENGL": RenderMode.OPENGL, "OPENGL3": RenderMode.OPENGL3}[render_mode_name]

    obs_config = ObservationConfig()
    obs_config.set_all(False)
    obs_config.gripper_pose = True
    obs_config.gripper_open = True
    obs_config.gripper_joint_positions = True
    obs_config.joint_positions = True

    def cc(rgb=True, depth=False):
        return CameraConfig(
            image_size=(IMG, IMG), rgb=rgb, depth=depth,
            point_cloud=False, mask=False, render_mode=rmode,
        )

    if multi_cam:
        for attr in ["front_camera", "overhead_camera",
                     "left_shoulder_camera", "right_shoulder_camera",
                     "wrist_camera"]:
            setattr(obs_config, attr, cc())
    elif overhead_depth_only:
        obs_config.front_camera = cc(rgb=True, depth=False)
        obs_config.overhead_camera = cc(rgb=False, depth=True)
    elif front_overhead:
        obs_config.front_camera = cc(rgb=True, depth=False)
        obs_config.overhead_camera = cc(rgb=True, depth=False)
    elif front_overhead_all_data:
        # Mimic lpwm-occ/rlbench_utils/dataset_generator.py: rgb+depth+pc+mask
        # all enabled. The texture-loading pass appears to depend on this.
        full = CameraConfig(
            image_size=(IMG, IMG), rgb=True, depth=True,
            point_cloud=True, mask=True, render_mode=rmode,
        )
        obs_config.front_camera = full
        obs_config.overhead_camera = full
    else:
        obs_config.front_camera = cc()

    am = MoveArmThenGripper(EndEffectorPoseViaPlanning(), Discrete())
    kw = dict(action_mode=am, obs_config=obs_config, headless=True)
    if dataset_root:
        kw["dataset_root"] = dataset_root
    env = Environment(**kw)
    env.launch()
    try:
        task = env.get_task(_resolve_task_class(TASK))
        task.set_variation(0)
        _, obs = task.reset()
        return np.asarray(obs.front_rgb, dtype=np.uint8)
    finally:
        env.shutdown()


def wrapper():
    from diffuser.envs.rlbench_dlp_wrapper import RLBenchDLPEnv
    env = RLBenchDLPEnv(
        task_name=TASK,
        dlp_encode_fn=lambda rgbs: {
            "tokens": np.zeros((40, 10), dtype=np.float32),
            "bg": np.zeros((8,), dtype=np.float32),
        },
        cams=["front", "overhead"],
        image_size=IMG,
        headless=True,
        episode_length=400,
        dataset_root=DATASET_ROOT,
    )
    try:
        env.reset(variation=0)
        frames, _ = env.pop_recorded_frames()
        return frames[0]
    finally:
        env.shutdown()


def generator_replica():
    """Literal replica of lpwm-occ/rlbench_utils/dataset_generator.py:161-202.

    5 cams via default ObservationConfig (so each cam gets its OWN CameraConfig
    object with rgb+depth+pc+mask all True), per-camera image_size, depth_in_meters=False,
    masks_as_one_channel=False, render_mode=OPENGL.
    """
    from rlbench import ObservationConfig
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.environment import Environment
    from pyrep.const import RenderMode
    from diffuser.envs.rlbench_dlp_wrapper import _resolve_task_class

    obs_config = ObservationConfig()
    obs_config.set_all(True)

    img_size = (IMG, IMG)
    for cam in ["right_shoulder_camera", "left_shoulder_camera",
                "overhead_camera", "wrist_camera", "front_camera"]:
        c = getattr(obs_config, cam)
        c.image_size = img_size
        c.depth_in_meters = False
        c.masks_as_one_channel = False
        c.render_mode = RenderMode.OPENGL

    am = MoveArmThenGripper(EndEffectorPoseViaPlanning(), Discrete())
    env = Environment(action_mode=am, obs_config=obs_config, headless=True)
    env.launch()
    try:
        task = env.get_task(_resolve_task_class(TASK))
        task.set_variation(0)
        _, obs = task.reset()
        # First frame straight from reset:
        first = np.asarray(obs.front_rgb, dtype=np.uint8)
        # Now step the sim a few times and re-grab via the scene's own
        # observation accessor — same path RLBench uses internally during
        # demo collection, so this matches what get_demos() would record.
        for _ in range(10):
            env._pyrep.step()
        warm_obs = task._scene.get_observation()
        warm = np.asarray(warm_obs.front_rgb, dtype=np.uint8)
        return first, warm
    finally:
        env.shutdown()


def main():
    print("[compare] RAW front-only, no dataset_root ...")
    a = raw(dataset_root=None, multi_cam=False)
    _save(a, "raw_front.png")

    print("[compare] RAW front-only, WITH dataset_root ...")
    b = raw(dataset_root=DATASET_ROOT, multi_cam=False)
    _save(b, "raw_datasetroot_front.png")

    print("[compare] RAW multi-cam (5 cams), no dataset_root ...")
    c = raw(dataset_root=None, multi_cam=True)
    _save(c, "raw_multicam_front.png")

    print("[compare] RAW front.rgb + overhead.depth (cam present, RGB not captured) ...")
    e = raw(dataset_root=None, multi_cam=False, overhead_depth_only=True)
    _save(e, "raw_overhead_depthonly_front.png")

    print("[compare] RAW front.rgb + overhead.rgb, OPENGL ...")
    f = raw(dataset_root=None, front_overhead=True, render_mode_name="OPENGL")
    _save(f, "raw_front_overhead_opengl.png")

    print("[compare] RAW front.rgb + overhead.rgb, OPENGL3 ...")
    g = raw(dataset_root=None, front_overhead=True, render_mode_name="OPENGL3")
    _save(g, "raw_front_overhead_opengl3.png")

    print("[compare] RAW front+overhead, ALL DATA TYPES (rgb+depth+pc+mask), OPENGL "
          "[mimics lpwm-occ dataset_generator] ...")
    h = raw(dataset_root=None, front_overhead_all_data=True, render_mode_name="OPENGL")
    _save(h, "raw_front_overhead_all_data_opengl.png")

    print("[compare] LITERAL REPLICA of lpwm-occ dataset_generator (5 cams, "
          "set_all(True), per-cam OPENGL) — first frame & after 10 sim steps ...")
    j_first, j_warm = generator_replica()
    _save(j_first, "raw_generator_replica_first.png")
    _save(j_warm, "raw_generator_replica_warm.png")

    print("[compare] WRAPPER (RLBenchDLPEnv, 2 cams + dataset_root) ...")
    d = wrapper()
    _save(d, "wrapper_front.png")


if __name__ == "__main__":
    main()
