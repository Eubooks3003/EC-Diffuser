#!/usr/bin/env python
"""
Compare RLBench front_rgb under RenderMode.OPENGL vs OPENGL3 on the same task/
variation/seed, and line them up against a training PNG from the DLP dataset.

Usage:
    cd /home/ellina/Desktop/EC-Diffuser
    conda activate ecdiffuser
    python diffuser/scripts/compare_render_modes.py \
        --task close_jar \
        --variation 0 \
        --image_size 128 \
        --out /home/ellina/Desktop/render_compare

Writes to <out>/:
    opengl_front.png         (wrapper-equivalent: RenderMode.OPENGL)
    opengl3_front.png        (RLBench default: RenderMode.OPENGL3)
    training_ref.png         (a matching PNG from the training set)
    side_by_side.png         (3 frames stitched horizontally with labels)
"""

import argparse
import os
import sys
import glob

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DIFFUSER_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
EC_DIFFUSER_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
for _p in [EC_DIFFUSER_ROOT, DIFFUSER_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def capture_one_frame(task_name: str, variation: int, image_size: int,
                      render_mode_name: str, seed: int) -> np.ndarray:
    """Launch RLBench, reset to (task, variation), grab front_rgb, shut down."""
    from rlbench import ObservationConfig, CameraConfig
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.environment import Environment
    from pyrep.const import RenderMode

    np.random.seed(seed)

    render_mode = {
        "OPENGL": RenderMode.OPENGL,
        "OPENGL3": RenderMode.OPENGL3,
    }[render_mode_name]

    obs_config = ObservationConfig()
    obs_config.set_all(False)
    obs_config.gripper_pose = True
    obs_config.gripper_open = True
    obs_config.joint_positions = True

    cc = CameraConfig(
        image_size=(image_size, image_size),
        rgb=True, depth=False, point_cloud=False, mask=False,
        render_mode=render_mode,
    )
    obs_config.front_camera = cc

    action_mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaPlanning(),
        gripper_action_mode=Discrete(),
    )

    env = Environment(action_mode=action_mode, obs_config=obs_config, headless=True)
    env.launch()
    try:
        from diffuser.envs.rlbench_dlp_wrapper import _resolve_task_class
        task_cls = _resolve_task_class(task_name)
        task = env.get_task(task_cls)
        var_count = task.variation_count()
        task.set_variation(int(variation) % var_count)
        _descriptions, rlbench_obs = task.reset()
        frame = np.asarray(rlbench_obs.front_rgb)
    finally:
        env.shutdown()

    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8) if frame.max() > 1.5 \
            else (np.clip(frame, 0, 1) * 255).astype(np.uint8)
    return frame


def find_training_png(task_name: str, image_size: int) -> str:
    """Return any training front_rgb PNG matching image_size, or '' if none."""
    roots = [
        f"/home/ellina/Desktop/data/rlbench_rgb/{task_name}/all_variations/episodes",
    ]
    for root in roots:
        pattern = os.path.join(root, "episode0", "front_rgb", "*.png")
        hits = sorted(glob.glob(pattern))
        if hits:
            for p in hits:
                try:
                    im = Image.open(p)
                    if im.size == (image_size, image_size):
                        return p
                except Exception:
                    continue
            return hits[0]
    return ""


def label_frame(arr: np.ndarray, label: str) -> np.ndarray:
    """Paste a white label strip across the top of a frame."""
    im = Image.fromarray(arr).convert("RGB")
    strip_h = max(14, im.height // 10)
    canvas = Image.new("RGB", (im.width, im.height + strip_h), (255, 255, 255))
    canvas.paste(im, (0, strip_h))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", strip_h - 4
        )
    except Exception:
        font = ImageFont.load_default()
    draw.text((4, 2), label, fill=(0, 0, 0), font=font)
    return np.asarray(canvas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="close_jar")
    ap.add_argument("--variation", type=int, default=0)
    ap.add_argument("--image_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="/home/ellina/Desktop/render_compare")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"[compare_render_modes] task={args.task} variation={args.variation} "
          f"size={args.image_size}")

    print("[compare_render_modes] capturing OPENGL...")
    opengl = capture_one_frame(
        args.task, args.variation, args.image_size, "OPENGL", args.seed
    )
    Image.fromarray(opengl).save(os.path.join(args.out, "opengl_front.png"))

    print("[compare_render_modes] capturing OPENGL3...")
    opengl3 = capture_one_frame(
        args.task, args.variation, args.image_size, "OPENGL3", args.seed
    )
    Image.fromarray(opengl3).save(os.path.join(args.out, "opengl3_front.png"))

    ref_path = find_training_png(args.task, args.image_size)
    if ref_path:
        ref = np.asarray(Image.open(ref_path).convert("RGB"))
        Image.fromarray(ref).save(os.path.join(args.out, "training_ref.png"))
        print(f"[compare_render_modes] training ref: {ref_path}")
    else:
        ref = np.zeros_like(opengl)
        print("[compare_render_modes] no training ref found")

    labeled = [
        label_frame(opengl, "OPENGL (current wrapper)"),
        label_frame(opengl3, "OPENGL3 (RLBench default)"),
        label_frame(ref, "TRAINING PNG"),
    ]
    target_h = max(x.shape[0] for x in labeled)
    resized = []
    for x in labeled:
        if x.shape[0] != target_h:
            im = Image.fromarray(x).resize(
                (int(x.shape[1] * target_h / x.shape[0]), target_h),
                Image.NEAREST,
            )
            x = np.asarray(im)
        resized.append(x)
    strip = np.concatenate(resized, axis=1)
    sbs_path = os.path.join(args.out, "side_by_side.png")
    Image.fromarray(strip).save(sbs_path)

    print(f"[compare_render_modes] wrote {sbs_path}")
    print("[compare_render_modes] mean RGB:")
    print(f"  OPENGL      : {opengl.reshape(-1,3).mean(axis=0).round(1)}")
    print(f"  OPENGL3     : {opengl3.reshape(-1,3).mean(axis=0).round(1)}")
    print(f"  training    : {ref.reshape(-1,3).mean(axis=0).round(1)}")


if __name__ == "__main__":
    main()
