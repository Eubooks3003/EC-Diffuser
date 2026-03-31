#!/usr/bin/env python
"""
Evaluation script for paper results.

Runs 50 rollouts with 3 different seeds and saves success rates.
Uses the same setup as training but with more rollouts.
Uses random_init=True for realistic evaluation (random environment initialization).

Usage:
    python eval_paper.py \
        --config mimicgen_hammer_cleanup_dlp \
        --mode 16C_dlp \
        --ckpt_path /path/to/ckpt/state_X_stepY.pt \
        --n_rollouts 50 \
        --seeds 42,123,456 \
        --output_dir ./eval_results \
        --save_videos \
        --video_episodes 5

Example for hammer_cleanup:
    python scripts/eval_paper.py \
        --config mimicgen_hammer_cleanup_dlp \
        --mode 16C_dlp \
        --ckpt_path /home/ubuntu/ellina/EC-Diffuser/data/hammer_cleanup/diffusion/mimicgen_stack/16C_dlp_adalnpint_relative_H16_T5_seed42/ckpt/state_0_step100000.pt \
        --n_rollouts 50 \
        --seeds 42,123,456 \
        --save_videos

Arguments:
    --config: Config file name (e.g., mimicgen_hammer_cleanup_dlp)
    --mode: Mode key in config (default: 16C_dlp)
    --ckpt_path: Path to checkpoint .pt file
    --n_rollouts: Number of rollouts per seed (default: 50)
    --seeds: Comma-separated list of seeds (default: 42,123,456)
    --output_dir: Output directory for results (default: alongside ckpt)
    --device: Device to use (default: cuda:0)
    --max_steps: Override max steps per episode (default: from config, typically 500)
    --save_videos: Enable video saving
    --video_episodes: Number of episodes to save videos for per seed (default: 5)
    --video_fps: Video FPS (default: 20)
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import argparse
import json
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm

# Add diffuser to path
DIFFUSER_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if DIFFUSER_ROOT not in sys.path:
    sys.path.insert(0, DIFFUSER_ROOT)

# Add EC-Diffuser root to path (for dlp_utils)
EC_DIFFUSER_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if EC_DIFFUSER_ROOT not in sys.path:
    sys.path.insert(0, EC_DIFFUSER_ROOT)

# Make lpwm-dev and lpwm-copy importable (siblings of EC-Diffuser)
_SCRIPT_DIR = os.path.dirname(__file__)
for _sibling in ("lpwm-dev", "lpwm-copy"):
    _p = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "..", "..", _sibling))
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)
    # Also check as sibling of EC-Diffuser directly
    _p2 = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "..", _sibling))
    if os.path.isdir(_p2) and _p2 not in sys.path:
        sys.path.append(_p2)


def build_dlp_3d_from_cfg(cfg, device, DLPClass):
    """Build a 3D (voxel) DLP model from config."""
    model = DLPClass(
        cdim=cfg["ch"],
        image_size=cfg["voxel_grid_whd"][0],
        normalize_rgb=cfg["normalize_rgb"],
        n_kp_per_patch=cfg["n_kp_per_patch"],
        patch_size=cfg["patch_size"],
        anchor_s=cfg["anchor_s"],
        n_kp_enc=cfg["n_kp_enc"],
        n_kp_prior=cfg["n_kp_prior"],
        pad_mode=cfg["pad_mode"],
        dropout=cfg["dropout"],
        features_dist=cfg.get("features_dist", "gauss"),
        learned_feature_dim=cfg["learned_feature_dim"],
        learned_bg_feature_dim=cfg.get("learned_bg_feature_dim", cfg["learned_feature_dim"]),
        n_fg_categories=cfg.get("n_fg_categories", 8),
        n_fg_classes=cfg.get("n_fg_classes", 4),
        n_bg_categories=cfg.get("n_bg_categories", 4),
        n_bg_classes=cfg.get("n_bg_classes", 4),
        scale_std=cfg["scale_std"],
        offset_std=cfg["offset_std"],
        obj_on_alpha=cfg["obj_on_alpha"],
        obj_on_beta=cfg["obj_on_beta"],
        obj_res_from_fc=cfg["obj_res_from_fc"],
        obj_ch_mult_prior=cfg.get("obj_ch_mult_prior", cfg["obj_ch_mult"]),
        obj_ch_mult=cfg["obj_ch_mult"],
        obj_base_ch=cfg["obj_base_ch"],
        obj_final_cnn_ch=cfg["obj_final_cnn_ch"],
        bg_res_from_fc=cfg["bg_res_from_fc"],
        bg_ch_mult=cfg["bg_ch_mult"],
        bg_base_ch=cfg["bg_base_ch"],
        bg_final_cnn_ch=cfg["bg_final_cnn_ch"],
        use_resblock=cfg["use_resblock"],
        num_res_blocks=cfg["num_res_blocks"],
        cnn_mid_blocks=cfg.get("cnn_mid_blocks", False),
        mlp_hidden_dim=cfg.get("mlp_hidden_dim", 256),
        pint_enc_layers=cfg["pint_enc_layers"],
        pint_enc_heads=cfg["pint_enc_heads"],
        timestep_horizon=1,
        separate_depth_features=cfg.get("separate_depth_features", False),
        depth_feature_dim=cfg.get("depth_feature_dim", 0),
        split_loss=cfg.get("split_loss", False),
        depth_loss_ratio=cfg.get("depth_loss_ratio", 1.0),
    ).to(device)
    model.eval()
    return model


def build_dlp_2d_from_cfg(cfg, device, DLPClass):
    """Build a 2D (image) DLP model from config."""
    model = DLPClass(
        cdim=cfg["ch"],
        image_size=cfg["image_size"],
        normalize_rgb=cfg.get("normalize_rgb", False),
        n_kp_per_patch=cfg["n_kp_per_patch"],
        patch_size=cfg["patch_size"],
        anchor_s=cfg["anchor_s"],
        n_kp_enc=cfg["n_kp_enc"],
        n_kp_prior=cfg["n_kp_prior"],
        pad_mode=cfg["pad_mode"],
        dropout=cfg.get("dropout", 0.0),
        features_dist=cfg.get("features_dist", "gauss"),
        learned_feature_dim=cfg["learned_feature_dim"],
        learned_bg_feature_dim=cfg.get("learned_bg_feature_dim", cfg["learned_feature_dim"]),
        n_fg_categories=cfg.get("n_fg_categories", 8),
        n_fg_classes=cfg.get("n_fg_classes", 4),
        n_bg_categories=cfg.get("n_bg_categories", 4),
        n_bg_classes=cfg.get("n_bg_classes", 4),
        scale_std=cfg["scale_std"],
        offset_std=cfg["offset_std"],
        obj_on_alpha=cfg["obj_on_alpha"],
        obj_on_beta=cfg["obj_on_beta"],
        obj_res_from_fc=cfg["obj_res_from_fc"],
        obj_ch_mult_prior=cfg.get("obj_ch_mult_prior", cfg["obj_ch_mult"]),
        obj_ch_mult=cfg["obj_ch_mult"],
        obj_base_ch=cfg["obj_base_ch"],
        obj_final_cnn_ch=cfg["obj_final_cnn_ch"],
        bg_res_from_fc=cfg["bg_res_from_fc"],
        bg_ch_mult=cfg["bg_ch_mult"],
        bg_base_ch=cfg["bg_base_ch"],
        bg_final_cnn_ch=cfg["bg_final_cnn_ch"],
        use_resblock=cfg["use_resblock"],
        num_res_blocks=cfg["num_res_blocks"],
        cnn_mid_blocks=cfg.get("cnn_mid_blocks", False),
        mlp_hidden_dim=cfg.get("mlp_hidden_dim", 256),
        pint_enc_layers=cfg["pint_enc_layers"],
        pint_enc_heads=cfg["pint_enc_heads"],
        timestep_horizon=1,
    ).to(device)
    model.eval()
    return model


def load_dlp_lpwm(dlp_cfg_path: str, dlp_ckpt_path: str, device: str,
                   dlp_ctor: str = "voxel_models:DLP"):
    """
    Load a DLP model (3D or 2D) based on dlp_ctor.

    dlp_ctor format: "module:ClassName"
      - "voxel_models:DLP"  -> 3D DLP from lpwm-dev
      - "models:DLP"        -> 2D DLP from lpwm-copy
    """
    from utils.util_func import get_config

    dev = torch.device(device)
    cfg = get_config(dlp_cfg_path)

    is_2d = "voxel" not in dlp_ctor.lower()

    if is_2d:
        from models import DLP as DLPClass
        model = build_dlp_2d_from_cfg(cfg, dev, DLPClass)
        model.load_state_dict(
            torch.load(dlp_ckpt_path, map_location=dev, weights_only=False)
        )
    else:
        from utils.log_utils import load_checkpoint
        from voxel_models import DLP as DLPClass
        model = build_dlp_3d_from_cfg(cfg, dev, DLPClass)
        _ = load_checkpoint(dlp_ckpt_path, model, None, None, map_location=dev)

    model.eval()
    return model, cfg


def _to_uint8(img):
    """Convert image to uint8 format for video saving."""
    img = np.asarray(img)
    if img.ndim == 4 and img.shape[0] == 1:
        img = img[0]
    # CHW -> HWC if needed
    if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        if img.max() <= 1.5:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _color_map(n=64):
    """Generate a distinct color map for keypoints (similar to lpwm-dev)."""
    import colorsys
    colors = []
    for i in range(n):
        hue = i / n
        rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.95)
        colors.append([int(c * 255) for c in rgb])
    return np.array(colors, dtype=np.uint8)


def _overlay_keypoints_on_frame(frame, toks, cam_idx=0, n_kp_per_view=None,
                                 kp_range=(-1, 1), radius=3, thickness=2):
    """
    Draw DLP keypoints on a camera frame.

    DLP convention (matching lpwm-dev plot_keypoints_on_image):
      z[:, 0] -> row (height axis)
      z[:, 1] -> col (width axis)
    cv2.circle takes (col, row) i.e. (x, y).

    Args:
        frame: (H, W, 3) uint8 image
        toks: (K_total, Dtok) token array from envw.last_toks.
              2D token layout: [z(2), z_scale(2), z_depth(1), obj_on(1), z_features(F)]
              3D token layout: [z(3), z_scale(3), z_depth(1), obj_on(1), z_features(F)]
              For 2D multiview: K_total = n_views * K_per_view, tokens concatenated.
        cam_idx: which camera view (0, 1, ...) to extract keypoints for
        n_kp_per_view: number of keypoints per view (set for 2D multiview).
        kp_range: range of keypoint coordinates (default [-1, 1])
        radius: circle radius
        thickness: circle thickness (-1 for filled)
    """
    import cv2

    if toks is None:
        return frame

    frame = frame.copy()
    h, w = frame.shape[:2]

    # Determine which tokens belong to this camera view
    if n_kp_per_view is not None and n_kp_per_view > 0:
        start = cam_idx * n_kp_per_view
        end = start + n_kp_per_view
        view_toks = toks[start:end]
        is_2d = True
    else:
        view_toks = toks
        is_2d = False

    # z positions: first 2 for 2D, first 3 for 3D (only use first 2 for image overlay)
    z = view_toks[:, :2]  # (K, 2) — z[:,0]=row, z[:,1]=col in kp_range

    # obj_on index depends on layout:
    #   2D: [z(2), scale(2), depth(1), obj_on(1), feat(F)] -> idx 5
    #   3D: [z(3), scale(3), depth(1), obj_on(1), feat(F)] -> idx 7
    obj_on_idx = 5 if is_2d else 7
    dtok = view_toks.shape[-1]
    if obj_on_idx < dtok:
        obj_on = view_toks[:, obj_on_idx]
    else:
        obj_on = np.ones(len(view_toks))

    # Convert from kp_range to pixel coordinates
    # z[:,0] is row -> maps to image height
    # z[:,1] is col -> maps to image width
    lo, hi = kp_range
    row_px = ((z[:, 0] - lo) / (hi - lo) * (h - 1))
    col_px = ((z[:, 1] - lo) / (hi - lo) * (w - 1))

    cmap = _color_map(len(view_toks))

    for i in range(len(view_toks)):
        alpha = float(np.clip(obj_on[i], 0, 1))
        if alpha < 0.05:
            continue  # skip inactive keypoints
        cx = int(np.clip(round(col_px[i]), 0, w - 1))
        cy = int(np.clip(round(row_px[i]), 0, h - 1))
        color = tuple(int(c) for c in cmap[i])
        cv2.circle(frame, (cx, cy), radius, color, thickness)

    return frame


@torch.no_grad()
def run_eval_rollouts(
    trainer,
    make_env_fn,
    dlp_model,
    calib_h5_path,
    goal_provider,
    task,
    n_episodes=50,
    max_steps=500,
    grid_dhw=(128, 128, 128),
    cams=("agentview", "sideview"),
    pixel_stride=1,
    exe_steps=8,
    seed=42,
    save_videos=False,
    video_dir=None,
    video_episodes=5,
    video_fps=20,
):
    """
    Run evaluation rollouts (simplified version of eval_mimicgen_rollouts).
    Saves one keypoint-overlay video per camera view per episode.
    Returns list of success booleans for each episode.
    """
    from diffuser.envs.mimicgen_dlp_wrapper import MimicGenDLPWrapper

    device = next(trainer.ema_model.parameters()).device
    dlp_model = dlp_model.to(device).eval()

    # Set seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Reset goal provider with this seed's shuffle
    if goal_provider is not None:
        goal_provider.reset_sampling(shuffle=True)

    successes = []
    returns = []
    lengths = []

    # Video saving setup
    if save_videos and video_dir is not None:
        os.makedirs(video_dir, exist_ok=True)
        try:
            import imageio
        except ImportError:
            print("[WARNING] imageio not installed, disabling video saving")
            save_videos = False

    print(f"[eval] seed={seed}, {n_episodes} eps, max_steps={max_steps}")

    # Create env and wrapper once, reuse across episodes
    env = make_env_fn()
    envw = MimicGenDLPWrapper(
        env=env,
        dlp_model=dlp_model,
        device=device,
        cams=cams,
        grid_dhw=grid_dhw,
        pixel_stride=pixel_stride,
        calib_h5_path=calib_h5_path,
        goal_provider=goal_provider,
        random_init=True,  # Always use random init for paper eval
        normalize_to_unit_cube=False,
        task=task,
    )

    # Get dimensions from trainer dataset (constant across episodes)
    a_dim = trainer.dataset.action_dim
    gripper_dim = getattr(trainer.dataset, 'gripper_dim', 0)
    bg_dim = getattr(trainer.dataset, 'bg_dim', 0)

    pbar = tqdm(range(n_episodes), desc=f"Seed {seed}", unit="ep")
    for ep in pbar:
        obs_vec = envw.reset()
        ep_ret = 0.0
        done = False

        # Video frame collection — one list per camera, keypoints overlaid
        record_this_episode = save_videos and ep < video_episodes
        # Record ALL cams (not just video_cams) so we get every training view
        rec_cams = list(cams)
        frames_per_cam = {cam: [] for cam in rec_cams} if record_this_episode else None

        # Determine n_kp_per_view for multiview 2D DLP
        n_kp_per_view = None
        if hasattr(envw, '_is_2d_dlp') and envw._is_2d_dlp() and len(cams) > 1:
            if hasattr(envw, 'last_toks') and envw.last_toks is not None:
                n_kp_per_view = envw.last_toks.shape[0] // len(cams)

        # Capture initial frame
        if record_this_episode and hasattr(envw, 'last_raw_obs'):
            raw_obs = envw.last_raw_obs
            toks = envw.last_toks if hasattr(envw, 'last_toks') else None
            for ci, cam in enumerate(rec_cams):
                k = f"{cam}_image"
                if k in raw_obs:
                    f_raw = _to_uint8(raw_obs[k])
                    frames_per_cam[cam].append(_overlay_keypoints_on_frame(
                        f_raw, toks, cam_idx=ci,
                        n_kp_per_view=n_kp_per_view,
                    ))

        # Action chunking setup
        action_buffer = None
        action_idx = 0

        t = 0
        while t < max_steps and not done:
            need_replan = (action_buffer is None) or (action_idx >= exe_steps)

            if need_replan:
                # Normalize observation
                obs_norm = trainer.dataset.normalizer.normalize(
                    obs_vec[None], "observations"
                )[0]

                # Build condition in correct order: [gripper, bg, obs]
                # This matches training.py which does: cond_parts = [gripper, bg, obs]
                cond_parts = []
                goal_parts = []

                # 1. Add gripper state if used (first in order)
                if gripper_dim > 0 and hasattr(envw, 'last_gripper_state'):
                    gripper_state = np.array(envw.last_gripper_state).flatten()[:gripper_dim]
                    gripper_norm = trainer.dataset.normalizer.normalize(
                        gripper_state[None], "gripper_state"
                    )[0]
                    cond_parts.append(gripper_norm)

                    goal_gripper = envw.goal_gripper_state if hasattr(envw, 'goal_gripper_state') else np.zeros(gripper_dim)
                    goal_gripper = np.array(goal_gripper).flatten()[:gripper_dim]
                    goal_gripper_norm = trainer.dataset.normalizer.normalize(
                        goal_gripper[None], "gripper_state"
                    )[0]
                    goal_parts.append(goal_gripper_norm)

                # 2. Add bg features if used (second in order)
                if bg_dim > 0 and hasattr(envw, 'last_bg_features'):
                    bg_features = np.array(envw.last_bg_features).flatten()[:bg_dim]
                    bg_norm = trainer.dataset.normalizer.normalize(
                        bg_features[None], "bg_features"
                    )[0]
                    cond_parts.append(bg_norm)

                    goal_bg = envw.goal_bg_features if hasattr(envw, 'goal_bg_features') else np.zeros(bg_dim)
                    goal_bg = np.array(goal_bg).flatten()[:bg_dim]
                    goal_bg_norm = trainer.dataset.normalizer.normalize(
                        goal_bg[None], "bg_features"
                    )[0]
                    goal_parts.append(goal_bg_norm)

                # 3. Add observations (last in order)
                cond_parts.append(obs_norm)
                goal_parts.append(np.zeros_like(obs_norm))

                # Concatenate all parts: [gripper, bg, obs]
                obs_norm = np.concatenate(cond_parts)
                goal_zeros = np.concatenate(goal_parts)

                # Build conditions — only condition on t=0 (matches GoalDataset training).
                # Zero goal at H-1 would force mean-state at every denoising step,
                # distorting predictions for a model that was never trained with it.
                cond = {
                    0: torch.from_numpy(obs_norm[None]).float().to(device),
                }

                # Sample trajectory from diffusion model
                sample = trainer.ema_model(cond, verbose=False)
                traj = sample.trajectories[0]  # (H, action_dim + obs_dim)
                action_buffer = traj[:, :a_dim].detach().cpu().numpy()
                action_idx = 0

            # Execute action
            a_norm = action_buffer[action_idx]
            a = trainer.dataset.normalizer.unnormalize(a_norm[None], "actions")[0]

            obs_vec, r, done, info = envw.step(a)

            # Debug: print info keys and success-related values (first episode only)
            if ep == 0 and t <= 3:
                succ_keys = {k: info[k] for k in info if 'success' in k.lower() or 'task' in k.lower() or 'done' in k.lower()}
                print(f"  [dbg] t={t} done={done} r={r:.4f} info_keys={list(info.keys())} succ_related={succ_keys}")

            # Capture frame for video
            if record_this_episode and hasattr(envw, 'last_raw_obs'):
                raw_obs = envw.last_raw_obs
                toks = envw.last_toks if hasattr(envw, 'last_toks') else None
                if n_kp_per_view is None and toks is not None:
                    if hasattr(envw, '_is_2d_dlp') and envw._is_2d_dlp() and len(cams) > 1:
                        n_kp_per_view = toks.shape[0] // len(cams)
                for ci, cam in enumerate(rec_cams):
                    k = f"{cam}_image"
                    if k in raw_obs:
                        f_raw = _to_uint8(raw_obs[k])
                        frames_per_cam[cam].append(_overlay_keypoints_on_frame(
                            f_raw, toks, cam_idx=ci,
                            n_kp_per_view=n_kp_per_view,
                        ))

            action_idx += 1
            ep_ret += float(r)
            t += 1

            # Check for success
            if info.get("success", False):
                done = True

        # Record results
        success = bool(info.get("success", False))
        successes.append(success)
        returns.append(ep_ret)
        lengths.append(t)
        pbar.set_postfix(sr=f"{np.mean(successes)*100:.0f}%", succ=sum(successes))

        # Save one kp-overlay video per camera
        if record_this_episode and video_dir is not None:
            import imageio
            status = "success" if success else "fail"
            for cam in rec_cams:
                cam_frames = frames_per_cam[cam]
                if cam_frames:
                    vpath = os.path.join(video_dir, f"seed{seed}_ep{ep:02d}_{status}_{cam}_kp.mp4")
                    try:
                        imageio.mimsave(vpath, cam_frames, fps=video_fps)
                    except Exception:
                        pass

    # Close environment after all episodes
    try:
        env.close()
    except:
        pass

    success_rate = float(np.mean(successes))
    print(f"[eval] Seed {seed}: success_rate={success_rate:.4f} ({sum(successes)}/{n_episodes})")

    return {
        "seed": seed,
        "n_episodes": n_episodes,
        "successes": successes,
        "success_rate": success_rate,
        "avg_return": float(np.mean(returns)),
        "avg_length": float(np.mean(lengths)),
    }


def main():
    parser = argparse.ArgumentParser(description="EC-Diffuser paper evaluation")
    parser.add_argument("--config", type=str, required=True,
                        help="Config file name (e.g., mimicgen_hammer_cleanup_dlp)")
    parser.add_argument("--mode", type=str, default="16C_dlp",
                        help="Mode key in config (default: 16C_dlp)")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--n_rollouts", type=int, default=50,
                        help="Number of rollouts per seed (default: 50)")
    parser.add_argument("--seeds", type=str, default="42,123,456",
                        help="Comma-separated list of seeds (default: 42,123,456)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results (default: alongside ckpt)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use (default: cuda:0)")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override max steps per episode (default: from config)")
    parser.add_argument("--save_videos", action="store_true",
                        help="Save videos of rollouts")
    parser.add_argument("--video_episodes", type=int, default=5,
                        help="Number of episodes to save videos for per seed (default: 5)")
    parser.add_argument("--video_fps", type=int, default=20,
                        help="Video FPS (default: 20)")
    args = parser.parse_args()

    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    print(f"\n{'='*60}")
    print(f"EC-Diffuser Paper Evaluation")
    print(f"{'='*60}")
    print(f"Config: {args.config}")
    print(f"Mode: {args.mode}")
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"N rollouts per seed: {args.n_rollouts}")
    print(f"Seeds: {seeds}")
    print(f"Device: {args.device}")
    print(f"Save videos: {args.save_videos}")
    if args.save_videos:
        print(f"Video episodes per seed: {args.video_episodes}")
    print(f"{'='*60}\n")

    # Import after setting up paths
    import diffuser.utils as utils
    from diffuser.utils.arrays import set_global_device

    # Load config
    config_module = __import__(f"config.{args.config}", fromlist=[args.config])
    mode_args = config_module.mode_to_args.get(args.mode, {})
    base_args = config_module.base["diffusion"]

    # Merge configs
    merged_args = {**base_args, **mode_args}

    # Create args namespace
    class Args:
        pass
    cfg = Args()
    for k, v in merged_args.items():
        setattr(cfg, k, v)

    # Override device
    cfg.device = args.device
    set_global_device(cfg.device)

    # Override max_steps if provided
    if args.max_steps is not None:
        cfg.mimicgen_max_steps = args.max_steps

    # Get paths from config
    dataset_path = getattr(cfg, 'override_dataset_path', None)
    calib_h5_path = getattr(cfg, 'calib_h5_path', None)
    dlp_ckpt = getattr(cfg, 'dlp_ckpt', None)
    dlp_cfg_path = getattr(cfg, 'dlp_cfg', None)

    if dataset_path is None:
        raise RuntimeError("Config must have 'override_dataset_path'")
    if calib_h5_path is None:
        raise RuntimeError("Config must have 'calib_h5_path'")
    if dlp_ckpt is None:
        raise RuntimeError("Config must have 'dlp_ckpt'")
    if dlp_cfg_path is None:
        raise RuntimeError("Config must have 'dlp_cfg'")

    # Load DLP model
    dlp_ctor = getattr(cfg, 'dlp_ctor', 'voxel_models:DLP')
    is_2d_dlp = "voxel" not in dlp_ctor.lower()
    print(f"Loading DLP model (dlp_ctor={dlp_ctor}, is_2d={is_2d_dlp})...")
    dlp_model, dlp_cfg = load_dlp_lpwm(dlp_cfg_path, dlp_ckpt, cfg.device, dlp_ctor=dlp_ctor)

    # For 2D DLP: override camera resolution to match preprocessing (84x84).
    # The 2D DLP was trained on 84x84 images (from preprocess_mimicgen_multiview.py).
    # Rendering at a different resolution (e.g. 256x256) and downsampling to the DLP's
    # image_size produces a different image distribution, causing OOD observations
    # and degraded policy performance.
    if is_2d_dlp:
        cfg.mimicgen_camera_width = 84
        cfg.mimicgen_camera_height = 84
        print(f"[2D DLP] Overriding camera resolution to 84x84 (matching preprocessing)")

    # Load dataset
    print("Loading dataset...")
    cfg.dataset_path = dataset_path
    cfg.savepath = os.path.dirname(args.ckpt_path).replace("/ckpt", "")

    dataset_config = utils.Config(
        cfg.loader,
        savepath=None,
        dataset_path=cfg.dataset_path,
        dataset_name=cfg.dataset,
        horizon=cfg.horizon,
        obs_only=getattr(cfg, 'obs_only', False),
        action_only=getattr(cfg, 'action_only', False),
        normalizer=cfg.normalizer,
        particle_normalizer=cfg.particle_normalizer,
        preprocess_fns=cfg.preprocess_fns,
        use_padding=cfg.use_padding,
        max_path_length=cfg.max_path_length,
        overfit=False,
        single_view=(getattr(cfg, 'input_type', 'dlp') == "dlp" and not cfg.multiview),
        action_z_scale=getattr(cfg, 'action_z_scale', 1.0),
        use_gripper_obs=getattr(cfg, 'use_gripper_obs', False),
        use_bg_obs=getattr(cfg, 'use_bg_obs', False),
    )
    dataset = dataset_config()

    # Build models
    print("Building diffusion model...")
    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim
    gripper_dim = getattr(dataset, 'gripper_dim', 0)
    bg_dim = getattr(dataset, 'bg_dim', 0)

    # The "singleview" models were trained on both views' particle tokens
    # (meta['K']=40 prevented buffer slicing). If the model expects more
    # particles than a single camera provides, update cfg so the env is
    # created with both cameras.
    n_kp_per_cam = getattr(dlp_cfg, 'n_kp_enc', 20)
    n_particles_expected = observation_dim // cfg.features_dim
    if n_particles_expected > n_kp_per_cam and len(getattr(cfg, 'mimicgen_cams', [])) == 1:
        print(f"[eval] Model expects {n_particles_expected} particles but 1 camera "
              f"produces {n_kp_per_cam}. Overriding to use both cameras.")
        cfg.mimicgen_cams = ["agentview", "sideview"]

    model_config = utils.Config(
        cfg.model,
        savepath=None,
        features_dim=cfg.features_dim,
        action_dim=action_dim,
        hidden_dim=cfg.hidden_dim,
        projection_dim=cfg.projection_dim,
        n_head=cfg.n_heads,
        n_layer=cfg.n_layers,
        dropout=cfg.dropout,
        block_size=cfg.horizon,
        positional_bias=cfg.positional_bias,
        max_particles=cfg.max_particles,
        multiview=cfg.multiview,
        device=cfg.device,
        gripper_dim=gripper_dim,
        bg_dim=bg_dim,
    )

    diffusion_config = utils.Config(
        cfg.diffusion,
        savepath=None,
        horizon=cfg.horizon,
        observation_dim=observation_dim,
        action_dim=action_dim,
        gripper_dim=gripper_dim,
        bg_dim=bg_dim,
        n_timesteps=cfg.n_diffusion_steps,
        loss_type=cfg.loss_type,
        clip_denoised=cfg.clip_denoised,
        predict_epsilon=cfg.predict_epsilon,
        action_weight=cfg.action_weight,
        loss_weights=cfg.loss_weights,
        loss_discount=cfg.loss_discount,
        device=cfg.device,
        obs_only=getattr(cfg, 'obs_only', False),
        action_only=getattr(cfg, 'action_only', False),
    )

    # Renderer is optional for evaluation (only needed for visualization during training)
    renderer = None
    try:
        render_config = utils.Config(
            cfg.renderer,
            savepath=None,
            env=None,
            particle_dim=cfg.features_dim,
        )
        renderer = render_config()
    except Exception:
        pass

    model = model_config()
    diffusion = diffusion_config(model)

    trainer_config = utils.Config(
        utils.Trainer,
        savepath=None,
        train_batch_size=cfg.batch_size,
        train_lr=cfg.learning_rate,
        gradient_accumulate_every=cfg.gradient_accumulate_every,
        ema_decay=cfg.ema_decay,
        sample_freq=cfg.sample_freq,
        save_freq=cfg.save_freq,
        label_freq=int(cfg.n_train_steps // cfg.n_saves),
        save_parallel=cfg.save_parallel,
        results_folder=cfg.savepath,
        bucket=cfg.bucket,
        n_reference=cfg.n_reference,
    )

    # Pass None for renderer if not available (renderer is only for training visualization)
    trainer = trainer_config(diffusion, dataset, renderer)

    # Load checkpoint
    ckpt_data = torch.load(args.ckpt_path, map_location=cfg.device)
    trainer.step = ckpt_data['step']
    trainer.model.load_state_dict(ckpt_data['model'])
    trainer.ema_model.load_state_dict(ckpt_data['ema'])
    print(f"Loaded checkpoint at step {trainer.step}")

    # Setup goal provider
    from diffuser.envs.mimicgen_dlp_wrapper import DatasetGoalProvider
    goal_provider = DatasetGoalProvider(dataset_path, shuffle=True)

    # Setup environment factory
    from diffuser.eval_utils import extract_mimicgen_task_name
    use_absolute_actions = getattr(cfg, 'use_absolute_actions', False)
    task = extract_mimicgen_task_name(calib_h5_path)
    print(f"Task: {task}")

    def make_env_fn():
        from diffuser.eval_utils import setup_mimicgen_env
        return setup_mimicgen_env(cfg, use_absolute_actions=use_absolute_actions)

    # Get eval parameters from config
    max_steps = getattr(cfg, 'mimicgen_max_steps', 500)
    exe_steps = getattr(cfg, 'exe_steps', 8)
    grid_dhw = getattr(cfg, 'mimicgen_grid_dhw', (128, 128, 128))
    cams = tuple(getattr(cfg, 'mimicgen_cams', ["agentview", "sideview"]))
    pixel_stride = getattr(cfg, 'mimicgen_pixel_stride', 1)

    # Setup video directory (include ckpt name so different steps don't overwrite)
    video_base_dir = None
    ckpt_name = os.path.basename(args.ckpt_path).replace(".pt", "")
    if args.save_videos:
        video_base_dir = os.path.join(
            args.output_dir if args.output_dir else os.path.join(os.path.dirname(args.ckpt_path), "eval_results"),
            "videos",
            ckpt_name,
        )
        os.makedirs(video_base_dir, exist_ok=True)

    # Run evaluation for each seed
    all_results = []
    for seed in seeds:

        video_dir = os.path.join(video_base_dir, f"seed_{seed}") if video_base_dir else None

        result = run_eval_rollouts(
            trainer=trainer,
            make_env_fn=make_env_fn,
            dlp_model=dlp_model,
            calib_h5_path=calib_h5_path,
            goal_provider=goal_provider,
            task=task,
            n_episodes=args.n_rollouts,
            max_steps=max_steps,
            grid_dhw=grid_dhw,
            cams=cams,
            pixel_stride=pixel_stride,
            exe_steps=exe_steps,
            seed=seed,
            save_videos=args.save_videos,
            video_dir=video_dir,
            video_episodes=args.video_episodes,
            video_fps=args.video_fps,
        )
        all_results.append(result)

    # Compute aggregate statistics
    all_success_rates = [r["success_rate"] for r in all_results]
    mean_success_rate = float(np.mean(all_success_rates))
    std_success_rate = float(np.std(all_success_rates))

    # Aggregate all individual successes
    all_successes = []
    for r in all_results:
        all_successes.extend(r["successes"])
    overall_success_rate = float(np.mean(all_successes))

    print(f"\nResults ({len(all_successes)} rollouts):")
    for r in all_results:
        print(f"  Seed {r['seed']}: {r['success_rate']*100:.1f}%")
    print(f"  Mean: {mean_success_rate*100:.1f}% +/- {std_success_rate*100:.1f}%")

    # Save results
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(args.ckpt_path), "eval_results")
    os.makedirs(output_dir, exist_ok=True)

    # Create results dict
    results = {
        "config": args.config,
        "mode": args.mode,
        "ckpt_path": args.ckpt_path,
        "ckpt_step": trainer.step,
        "n_rollouts_per_seed": args.n_rollouts,
        "seeds": seeds,
        "task": task,
        "max_steps": max_steps,
        "exe_steps": exe_steps,
        "random_init": True,
        "timestamp": datetime.now().isoformat(),
        "per_seed_results": all_results,
        "mean_success_rate": mean_success_rate,
        "std_success_rate": std_success_rate,
        "overall_success_rate": overall_success_rate,
        "total_rollouts": len(all_successes),
    }

    # Save JSON
    output_file = os.path.join(output_dir, f"eval_{ckpt_name}_seeds{'_'.join(map(str, seeds))}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: x if not isinstance(x, np.ndarray) else x.tolist())

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
