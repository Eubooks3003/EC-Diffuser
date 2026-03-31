#!/usr/bin/env python
"""
Visualize the diffuser's imagined (predicted) states during evaluation rollouts.

At each replan step, the diffusion model generates a full trajectory of future
particle states. This script decodes those predicted particle tokens through the
DLP model and saves side-by-side comparisons of:
  - The ACTUAL current observation (what the robot sees)
  - The IMAGINED future states (what the diffuser predicts will happen)

Supports both 3D (voxel) and 2D (image) DLP models.

Usage:
    python visualize_imagined_states.py \
        --config mimicgen_hammer_cleanup_dlp \
        --mode 16C_dlp \
        --ckpt_path /path/to/checkpoint.pt \
        --output_dir ./imagined_states_vis \
        --max_steps 300 \
        --seed 42 \
        --n_episodes 1 \
        --save_horizon_frames   # save decoded images for each horizon step
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import argparse
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

# ---------------------------------------------------------------------------
# Path setup (same as eval_paper.py)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DIFFUSER_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
EC_DIFFUSER_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

for p in [EC_DIFFUSER_ROOT, DIFFUSER_ROOT]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

for _sibling in ("lpwm-dev", "lpwm-copy"):
    _p = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", "..", _sibling))
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)
    _p2 = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", _sibling))
    if os.path.isdir(_p2) and _p2 not in sys.path:
        sys.path.append(_p2)


# ---------------------------------------------------------------------------
# DLP builders (reused from eval_paper.py)
# ---------------------------------------------------------------------------
def build_dlp_3d_from_cfg(cfg, device, DLPClass):
    model = DLPClass(
        cdim=cfg["ch"], image_size=cfg["voxel_grid_whd"][0],
        normalize_rgb=cfg["normalize_rgb"], n_kp_per_patch=cfg["n_kp_per_patch"],
        patch_size=cfg["patch_size"], anchor_s=cfg["anchor_s"],
        n_kp_enc=cfg["n_kp_enc"], n_kp_prior=cfg["n_kp_prior"],
        pad_mode=cfg["pad_mode"], dropout=cfg["dropout"],
        features_dist=cfg.get("features_dist", "gauss"),
        learned_feature_dim=cfg["learned_feature_dim"],
        learned_bg_feature_dim=cfg.get("learned_bg_feature_dim", cfg["learned_feature_dim"]),
        n_fg_categories=cfg.get("n_fg_categories", 8),
        n_fg_classes=cfg.get("n_fg_classes", 4),
        n_bg_categories=cfg.get("n_bg_categories", 4),
        n_bg_classes=cfg.get("n_bg_classes", 4),
        scale_std=cfg["scale_std"], offset_std=cfg["offset_std"],
        obj_on_alpha=cfg["obj_on_alpha"], obj_on_beta=cfg["obj_on_beta"],
        obj_res_from_fc=cfg["obj_res_from_fc"],
        obj_ch_mult_prior=cfg.get("obj_ch_mult_prior", cfg["obj_ch_mult"]),
        obj_ch_mult=cfg["obj_ch_mult"], obj_base_ch=cfg["obj_base_ch"],
        obj_final_cnn_ch=cfg["obj_final_cnn_ch"],
        bg_res_from_fc=cfg["bg_res_from_fc"], bg_ch_mult=cfg["bg_ch_mult"],
        bg_base_ch=cfg["bg_base_ch"], bg_final_cnn_ch=cfg["bg_final_cnn_ch"],
        use_resblock=cfg["use_resblock"], num_res_blocks=cfg["num_res_blocks"],
        cnn_mid_blocks=cfg.get("cnn_mid_blocks", False),
        mlp_hidden_dim=cfg.get("mlp_hidden_dim", 256),
        pint_enc_layers=cfg["pint_enc_layers"], pint_enc_heads=cfg["pint_enc_heads"],
        timestep_horizon=1,
        separate_depth_features=cfg.get("separate_depth_features", False),
        depth_feature_dim=cfg.get("depth_feature_dim", 0),
        split_loss=cfg.get("split_loss", False),
        depth_loss_ratio=cfg.get("depth_loss_ratio", 1.0),
    ).to(device)
    model.eval()
    return model


def build_dlp_2d_from_cfg(cfg, device, DLPClass):
    model = DLPClass(
        cdim=cfg["ch"], image_size=cfg["image_size"],
        normalize_rgb=cfg.get("normalize_rgb", False),
        n_kp_per_patch=cfg["n_kp_per_patch"], patch_size=cfg["patch_size"],
        anchor_s=cfg["anchor_s"], n_kp_enc=cfg["n_kp_enc"],
        n_kp_prior=cfg["n_kp_prior"], pad_mode=cfg["pad_mode"],
        dropout=cfg.get("dropout", 0.0),
        features_dist=cfg.get("features_dist", "gauss"),
        learned_feature_dim=cfg["learned_feature_dim"],
        learned_bg_feature_dim=cfg.get("learned_bg_feature_dim", cfg["learned_feature_dim"]),
        n_fg_categories=cfg.get("n_fg_categories", 8),
        n_fg_classes=cfg.get("n_fg_classes", 4),
        n_bg_categories=cfg.get("n_bg_categories", 4),
        n_bg_classes=cfg.get("n_bg_classes", 4),
        scale_std=cfg["scale_std"], offset_std=cfg["offset_std"],
        obj_on_alpha=cfg["obj_on_alpha"], obj_on_beta=cfg["obj_on_beta"],
        obj_res_from_fc=cfg["obj_res_from_fc"],
        obj_ch_mult_prior=cfg.get("obj_ch_mult_prior", cfg["obj_ch_mult"]),
        obj_ch_mult=cfg["obj_ch_mult"], obj_base_ch=cfg["obj_base_ch"],
        obj_final_cnn_ch=cfg["obj_final_cnn_ch"],
        bg_res_from_fc=cfg["bg_res_from_fc"], bg_ch_mult=cfg["bg_ch_mult"],
        bg_base_ch=cfg["bg_base_ch"], bg_final_cnn_ch=cfg["bg_final_cnn_ch"],
        use_resblock=cfg["use_resblock"], num_res_blocks=cfg["num_res_blocks"],
        cnn_mid_blocks=cfg.get("cnn_mid_blocks", False),
        mlp_hidden_dim=cfg.get("mlp_hidden_dim", 256),
        pint_enc_layers=cfg["pint_enc_layers"], pint_enc_heads=cfg["pint_enc_heads"],
        timestep_horizon=1,
    ).to(device)
    model.eval()
    return model


def load_dlp_lpwm(dlp_cfg_path, dlp_ckpt_path, device, dlp_ctor="voxel_models:DLP"):
    from utils.util_func import get_config
    dev = torch.device(device)
    cfg = get_config(dlp_cfg_path)
    is_2d = "voxel" not in dlp_ctor.lower()
    if is_2d:
        from models import DLP as DLPClass
        model = build_dlp_2d_from_cfg(cfg, dev, DLPClass)
        model.load_state_dict(torch.load(dlp_ckpt_path, map_location=dev, weights_only=False))
    else:
        from utils.log_utils import load_checkpoint
        from voxel_models import DLP as DLPClass
        model = build_dlp_3d_from_cfg(cfg, dev, DLPClass)
        _ = load_checkpoint(dlp_ckpt_path, model, None, None, map_location=dev)
    model.eval()
    return model, cfg


# ---------------------------------------------------------------------------
# Core visualization: decode particle tokens to images
# ---------------------------------------------------------------------------
@torch.no_grad()
def _decode_single_view_2d(dlp_model, toks_np, bg_np, device):
    """
    Decode one view of 2D DLP tokens to a reconstructed image.

    Token layout (from MimicGenDLPWrapper._encode_tokens_2d):
        [z(2), z_scale(2), z_depth(1), obj_on(1), z_features(F)]

    NOTE: get_recon_from_dlps assumes obj_on is LAST, but the wrapper
    packs obj_on at index 5 with features after it. We unpack manually
    to match the actual layout.
    """
    toks_t = torch.from_numpy(toks_np).float().unsqueeze(0).to(device)  # (1, K, Dtok)

    z = toks_t[..., 0:2]          # (1, K, 2)
    z_scale = toks_t[..., 2:4]    # (1, K, 2)
    z_depth = toks_t[..., 4:5]    # (1, K, 1)
    obj_on = toks_t[..., 5:6]     # (1, K, 1)
    z_feat = toks_t[..., 6:]      # (1, K, F)

    z_bg = None
    if bg_np is not None:
        z_bg = torch.from_numpy(np.asarray(bg_np).flatten()).float().to(device)
        z_bg = z_bg.unsqueeze(0)  # (1, bg_dim)

    # lpwm-copy DLP.decode_all signature:
    #   (z, z_scale, z_features, obj_on, z_depth, z_bg_features, z_ctx, warmup)
    dec = dlp_model.decode_all(
        z, z_scale, z_feat, obj_on, z_depth, z_bg, None, warmup=False
    )

    recon = dec['rec']
    recon = recon.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
    recon = (np.clip(recon, 0, 1) * 255).astype(np.uint8)
    return recon


@torch.no_grad()
def decode_tokens_to_image_2d(dlp_model, toks, bg_features, device, multiview=False):
    """
    Decode 2D DLP particle tokens to a reconstructed image.

    For multiview (e.g. agentview + sideview):
      - toks: (K_total, Dtok) where K_total = K_per_view * n_views
      - bg_features: (bg_dim_total,) where bg_dim_total = bg_per_view * n_views
      Each view is decoded separately and concatenated horizontally.

    Args:
        dlp_model: 2D DLP model
        toks: (K, Dtok) particle tokens (numpy)
        bg_features: background features (numpy) or None
        device: torch device
        multiview: if True, split particles and bg into 2 views

    Returns:
        (H, W, 3) uint8 image (or (H, W*2, 3) for multiview)
    """
    if multiview:
        K_total = toks.shape[0]
        K_per_view = K_total // 2
        front_toks = toks[:K_per_view]
        side_toks = toks[K_per_view:]

        front_bg, side_bg = None, None
        if bg_features is not None:
            bg_flat = np.asarray(bg_features).flatten()
            bg_per_view = len(bg_flat) // 2
            front_bg = bg_flat[:bg_per_view]
            side_bg = bg_flat[bg_per_view:]

        front_img = _decode_single_view_2d(dlp_model, front_toks, front_bg, device)
        side_img = _decode_single_view_2d(dlp_model, side_toks, side_bg, device)
        return np.concatenate([front_img, side_img], axis=1)
    else:
        return _decode_single_view_2d(dlp_model, toks, bg_features, device)


@torch.no_grad()
def decode_tokens_to_voxel_3d(dlp_model, toks, bg_features, device):
    """
    Decode 3D DLP particle tokens to a voxel volume.

    Args:
        dlp_model: 3D DLP model
        toks: (K, Dtok) particle tokens (numpy)
        bg_features: background features (numpy) or None
        device: torch device

    Returns:
        dict with:
            'rec_rgb': (3, D, H, W) float tensor - full reconstruction
            'fg_only': (3, D, H, W) float tensor - foreground only
            'kp_positions': (K, 3) numpy - keypoint positions in [-1, 1]
            'obj_on': (K,) numpy - object activation values
    """
    toks_t = torch.from_numpy(toks).float().to(device)
    K, Dtok = toks_t.shape

    # Token layout: [z(3), z_scale(3), z_depth(1), obj_on(1), z_features(F)]
    z = toks_t[:, 0:3].unsqueeze(0).unsqueeze(0)        # (1, 1, K, 3)
    z_scale = toks_t[:, 3:6].unsqueeze(0).unsqueeze(0)  # (1, 1, K, 3)
    z_depth = toks_t[:, 6:7].unsqueeze(0).unsqueeze(0)  # (1, 1, K, 1)
    obj_on = toks_t[:, 7:8].unsqueeze(0).unsqueeze(0)   # (1, 1, K, 1)
    z_feat = toks_t[:, 8:].unsqueeze(0).unsqueeze(0)    # (1, 1, K, F)

    z_bg = None
    if bg_features is not None:
        bg_np = np.asarray(bg_features).flatten()
        z_bg = torch.from_numpy(bg_np).float().to(device)
        z_bg = z_bg.unsqueeze(0).unsqueeze(0)  # (1, 1, bg_dim)

    dec = dlp_model.decode_all(z, z_scale, z_feat, obj_on, z_depth, z_bg, None, warmup=False)

    kp_pos = toks[:, 0:3]   # (K, 3) numpy
    obj_on_np = toks[:, 7]  # (K,) numpy

    return {
        'rec_rgb': dec['rec_rgb'][0, :3].cpu(),       # (3, D, H, W)
        'fg_only': dec['dec_objects_trans'][0, :3].cpu(),  # (3, D, H, W)
        'kp_positions': kp_pos,
        'obj_on': obj_on_np,
    }


def voxel_to_2d_projections(vol, axis_names=('front_XY', 'top_XZ', 'side_YZ')):
    """
    Project a (3, D, H, W) voxel volume to 2D images via max-intensity projection.

    Returns:
        dict mapping axis_name -> (H_proj, W_proj, 3) uint8 image
    """
    vol_np = vol.numpy() if torch.is_tensor(vol) else np.asarray(vol)
    # vol_np: (3, D, H, W)  — D=depth, H=height, W=width
    projections = {}

    # Front view: max over depth axis (axis=1) -> (3, H, W)
    front = vol_np.max(axis=1)
    projections[axis_names[0]] = _vol_slice_to_uint8(front)

    # Top view: max over height axis (axis=2) -> (3, D, W)
    top = vol_np.max(axis=2)
    projections[axis_names[1]] = _vol_slice_to_uint8(top)

    # Side view: max over width axis (axis=3) -> (3, D, H)
    side = vol_np.max(axis=3)
    projections[axis_names[2]] = _vol_slice_to_uint8(side)

    return projections


def _vol_slice_to_uint8(img_chw):
    """Convert (3, H, W) float to (H, W, 3) uint8."""
    img = np.transpose(img_chw, (1, 2, 0))
    if img.max() <= 1.5:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    else:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


# ---------------------------------------------------------------------------
# Visualization panel: actual obs vs imagined trajectory
# ---------------------------------------------------------------------------
def save_replan_panel(
    actual_img,
    imagined_imgs,
    horizon_indices,
    save_path,
    title="",
    is_3d=False,
):
    """
    Save a panel comparing actual observation to imagined future states.

    Args:
        actual_img: (H, W, 3) uint8 - current actual observation
        imagined_imgs: list of (H, W, 3) uint8 - decoded imagined states
        horizon_indices: list of int - which horizon steps these correspond to
        save_path: path to save the panel image
        title: title for the panel
        is_3d: if True, imagined_imgs are projections (dict per step)
    """
    n_imagined = len(imagined_imgs)
    n_cols = 1 + n_imagined
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    # Actual observation
    axes[0].imshow(actual_img)
    axes[0].set_title("Actual (t=now)", fontsize=10, fontweight='bold')
    axes[0].axis('off')

    # Imagined future states
    for i, (img, h_idx) in enumerate(zip(imagined_imgs, horizon_indices)):
        axes[i + 1].imshow(img)
        axes[i + 1].set_title(f"Imagined h={h_idx}", fontsize=10)
        axes[i + 1].axis('off')

    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_3d_replan_panel(
    actual_projs,
    imagined_projs_list,
    horizon_indices,
    save_path,
    title="",
):
    """
    Save a multi-row panel for 3D: rows = projection axes, cols = timesteps.

    Args:
        actual_projs: dict mapping view_name -> (H, W, 3) uint8
        imagined_projs_list: list of dicts (one per horizon step)
        horizon_indices: list of int
        save_path: path to save
        title: panel title
    """
    view_names = list(actual_projs.keys())
    n_views = len(view_names)
    n_cols = 1 + len(imagined_projs_list)

    fig, axes = plt.subplots(n_views, n_cols, figsize=(4 * n_cols, 4 * n_views))
    if n_views == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for row, view_name in enumerate(view_names):
        # Actual
        axes[row, 0].imshow(actual_projs[view_name])
        if row == 0:
            axes[row, 0].set_title("Actual (t=now)", fontsize=10, fontweight='bold')
        axes[row, 0].set_ylabel(view_name, fontsize=9)
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

        # Imagined
        for col, (projs, h_idx) in enumerate(zip(imagined_projs_list, horizon_indices)):
            axes[row, col + 1].imshow(projs[view_name])
            if row == 0:
                axes[row, col + 1].set_title(f"Imagined h={h_idx}", fontsize=10)
            axes[row, col + 1].set_xticks([])
            axes[row, col + 1].set_yticks([])

    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_keypoint_trajectory_plot(
    kp_trajectory,
    obj_on_trajectory,
    horizon_indices,
    save_path,
    title="Keypoint Trajectory (Imagined)",
):
    """
    Plot the 3D trajectory of keypoints across the imagined horizon.

    Args:
        kp_trajectory: list of (K, 3) arrays - keypoint positions per horizon step
        obj_on_trajectory: list of (K,) arrays - obj_on values per step
        horizon_indices: list of int
        save_path: path to save
        title: plot title
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    n_kp = kp_trajectory[0].shape[0]
    import colorsys
    colors = []
    for i in range(n_kp):
        hue = i / max(n_kp, 1)
        rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.95)
        colors.append(rgb)

    for k in range(n_kp):
        xs = [kp_trajectory[t][k, 0] for t in range(len(kp_trajectory))]
        ys = [kp_trajectory[t][k, 1] for t in range(len(kp_trajectory))]
        zs = [kp_trajectory[t][k, 2] for t in range(len(kp_trajectory))]

        # Only plot if the keypoint is active (obj_on > 0.5) in at least one step
        avg_on = np.mean([obj_on_trajectory[t][k] for t in range(len(obj_on_trajectory))])
        if avg_on < 0.3:
            continue

        ax.plot(xs, ys, zs, color=colors[k], linewidth=1.5, alpha=0.7)
        # Mark start and end
        ax.scatter([xs[0]], [ys[0]], [zs[0]], color=colors[k], s=30, marker='o')
        ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], color=colors[k], s=30, marker='^')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main evaluation with imagined-state visualization
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_with_imagined_vis(
    trainer, dlp_model, envw, dataset, device,
    max_steps=300, exe_steps=8, seed=42,
    output_dir="./imagined_vis",
    save_horizon_frames=False,
    max_replans_to_log=10,
    is_3d=True,
    multiview=False,
):
    """
    Run one episode and save imagined-state visualizations at each replan.

    Returns: dict with episode results
    """
    os.makedirs(output_dir, exist_ok=True)

    np.random.seed(seed)
    torch.manual_seed(seed)

    a_dim = dataset.action_dim
    gripper_dim = getattr(dataset, 'gripper_dim', 0)
    bg_dim = getattr(dataset, 'bg_dim', 0)
    obs_start_idx = a_dim + gripper_dim + bg_dim
    H = dataset.horizon

    obs_vec = envw.reset()
    K = envw.last_toks.shape[0]
    Dtok = envw.last_toks.shape[1]
    print(f"Obs: K={K}, Dtok={Dtok}, obs_dim={obs_vec.shape}, H={H}")

    action_buffer = None
    action_idx = 0
    plan_idx = 0
    t = 0
    done = False

    use_gripper_obs = getattr(dataset, 'use_gripper_obs', False)
    use_bg_obs = getattr(dataset, 'use_bg_obs', False)

    results = {"replans": [], "success": False, "length": 0}

    while t < max_steps and not done:
        need_replan = (action_buffer is None) or (action_idx >= exe_steps)

        if need_replan:
            # Build condition
            cond_parts = []
            if use_gripper_obs and gripper_dim > 0:
                gripper_cond = envw.get_gripper_cond(horizon=H)
                if gripper_cond is not None and 0 in gripper_cond:
                    g_norm = dataset.normalizer.normalize(gripper_cond[0][None], "gripper_state")[0]
                    cond_parts.append(torch.from_numpy(g_norm).float().to(device))
            if use_bg_obs and bg_dim > 0:
                bg_cond = envw.get_bg_cond(horizon=H)
                if bg_cond is not None and 0 in bg_cond:
                    b_norm = dataset.normalizer.normalize(bg_cond[0][None], "bg_features")[0]
                    cond_parts.append(torch.from_numpy(b_norm).float().to(device))
            obs_norm = dataset.normalizer.normalize(obs_vec[None], "observations")[0]
            cond_parts.append(torch.from_numpy(obs_norm).float().to(device))

            cond_0 = torch.cat(cond_parts, dim=-1)[None, :]
            cond = {0: cond_0}

            # Sample trajectory from diffusion model
            sample = trainer.ema_model(cond, verbose=False)
            traj = sample.trajectories[0]  # (H, transition_dim)
            action_buffer = traj[:, :a_dim].detach().cpu().numpy()
            action_idx = 0

            # ============================================================
            # VISUALIZE IMAGINED STATES at this replan step
            # ============================================================
            if plan_idx < max_replans_to_log:
                replan_dir = os.path.join(output_dir, f"replan_{plan_idx:03d}_t{t:04d}")
                os.makedirs(replan_dir, exist_ok=True)

                # 1) Decode the ACTUAL current observation
                actual_toks = envw.last_toks  # (K, Dtok)
                actual_bg = envw.last_bg_features

                # 2) Extract predicted observations from the trajectory
                pred_obs_norm = traj[:, obs_start_idx:].detach().cpu().numpy()
                pred_obs = dataset.normalizer.unnormalize(pred_obs_norm, "observations")

                pred_bg_all = None
                if use_bg_obs and bg_dim > 0:
                    bg_s = a_dim + gripper_dim
                    bg_e = bg_s + bg_dim
                    pred_bg_norm = traj[:, bg_s:bg_e].detach().cpu().numpy()
                    pred_bg_all = dataset.normalizer.unnormalize(pred_bg_norm, "bg_features")

                # Choose horizon steps to visualize
                if H <= 6:
                    h_indices = list(range(H))
                else:
                    h_indices = [0, H // 4, H // 2, 3 * H // 4, H - 1]

                if is_3d:
                    # --- 3D visualization ---
                    # Decode actual
                    actual_dec = decode_tokens_to_voxel_3d(dlp_model, actual_toks, actual_bg, device)
                    actual_projs = voxel_to_2d_projections(actual_dec['rec_rgb'])

                    imagined_projs_list = []
                    kp_trajectory = []
                    obj_on_trajectory = []

                    for h_idx in (list(range(H)) if save_horizon_frames else h_indices):
                        pred_toks = pred_obs[h_idx].reshape(K, Dtok)
                        bg_h = pred_bg_all[h_idx] if pred_bg_all is not None else actual_bg
                        pred_dec = decode_tokens_to_voxel_3d(dlp_model, pred_toks, bg_h, device)

                        kp_trajectory.append(pred_dec['kp_positions'])
                        obj_on_trajectory.append(pred_dec['obj_on'])

                        if h_idx in h_indices:
                            pred_projs = voxel_to_2d_projections(pred_dec['rec_rgb'])
                            imagined_projs_list.append(pred_projs)

                        if save_horizon_frames:
                            frame_dir = os.path.join(replan_dir, "frames")
                            os.makedirs(frame_dir, exist_ok=True)
                            for view_name, img in voxel_to_2d_projections(pred_dec['rec_rgb']).items():
                                frame_path = os.path.join(frame_dir, f"h{h_idx:02d}_{view_name}.png")
                                plt.imsave(frame_path, img)

                    # Save comparison panel
                    save_3d_replan_panel(
                        actual_projs, imagined_projs_list, h_indices,
                        os.path.join(replan_dir, "actual_vs_imagined.png"),
                        title=f"Replan {plan_idx} at t={t}",
                    )

                    # Save keypoint trajectory plot
                    if len(kp_trajectory) > 1:
                        save_keypoint_trajectory_plot(
                            kp_trajectory, obj_on_trajectory,
                            list(range(len(kp_trajectory))),
                            os.path.join(replan_dir, "kp_trajectory_3d.png"),
                            title=f"KP Trajectory (Replan {plan_idx}, t={t})",
                        )

                else:
                    # --- 2D visualization ---
                    actual_img = decode_tokens_to_image_2d(
                        dlp_model, actual_toks, actual_bg, device, multiview=multiview)

                    imagined_imgs = []
                    for h_idx in h_indices:
                        pred_toks = pred_obs[h_idx].reshape(K, Dtok)
                        bg_h = pred_bg_all[h_idx] if pred_bg_all is not None else actual_bg
                        pred_img = decode_tokens_to_image_2d(
                            dlp_model, pred_toks, bg_h, device, multiview=multiview)
                        imagined_imgs.append(pred_img)

                    save_replan_panel(
                        actual_img, imagined_imgs, h_indices,
                        os.path.join(replan_dir, "actual_vs_imagined.png"),
                        title=f"Replan {plan_idx} at t={t}",
                    )

                # Save raw predicted tokens (for later analysis)
                np.savez_compressed(
                    os.path.join(replan_dir, "predicted_tokens.npz"),
                    pred_obs=pred_obs,
                    actual_toks=actual_toks,
                    horizon_indices=np.array(h_indices),
                    timestep=t,
                    plan_idx=plan_idx,
                )

                replan_info = {
                    "plan_idx": plan_idx, "timestep": t,
                    "output_dir": replan_dir,
                }
                results["replans"].append(replan_info)
                print(f"  [replan {plan_idx}] t={t} -> saved to {replan_dir}")

            plan_idx += 1

        # Execute action
        a_norm = action_buffer[action_idx]
        a = dataset.normalizer.unnormalize(a_norm[None], "actions")[0]
        obs_vec, r, done, info = envw.step(a)
        action_idx += 1
        t += 1

        if info.get("success", False):
            print(f"  SUCCESS at t={t}")
            done = True

    results["success"] = bool(info.get("success", False))
    results["length"] = t
    results["n_replans"] = plan_idx

    # Save results summary
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Visualize diffuser imagined states")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", type=str, default="16C_dlp")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--n_episodes", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="./imagined_states_vis")
    parser.add_argument("--save_horizon_frames", action="store_true",
                        help="Save decoded image for every horizon step (not just selected)")
    parser.add_argument("--max_replans_to_log", type=int, default=20,
                        help="Max number of replan steps to visualize per episode")
    parser.add_argument("--random_init", action="store_true", default=True)
    parser.add_argument("--no_random_init", dest="random_init", action="store_false")
    args = parser.parse_args()

    import diffuser.utils as utils
    from diffuser.utils.arrays import set_global_device

    # Load config
    config_module = __import__(f"config.{args.config}", fromlist=[args.config])
    mode_args = config_module.mode_to_args.get(args.mode, {})
    base_args = config_module.base["diffusion"]
    merged_args = {**base_args, **mode_args}

    class Args:
        pass
    cfg = Args()
    for k, v in merged_args.items():
        setattr(cfg, k, v)
    cfg.device = args.device
    set_global_device(cfg.device)

    # Load DLP
    dlp_ctor = getattr(cfg, 'dlp_ctor', 'voxel_models:DLP')
    is_3d = "voxel" in dlp_ctor.lower()
    print(f"Loading DLP model (ctor={dlp_ctor}, is_3d={is_3d})...")
    dlp_model, _ = load_dlp_lpwm(
        getattr(cfg, 'dlp_cfg'), getattr(cfg, 'dlp_ckpt'), cfg.device,
        dlp_ctor=dlp_ctor
    )

    # Load dataset + diffusion model (same as eval_paper.py)
    dataset_path = getattr(cfg, 'override_dataset_path')
    cfg.dataset_path = dataset_path
    cfg.savepath = os.path.dirname(args.ckpt_path).replace("/ckpt", "")

    dataset_config = utils.Config(
        cfg.loader, savepath=None, dataset_path=cfg.dataset_path,
        dataset_name=cfg.dataset, horizon=cfg.horizon,
        obs_only=getattr(cfg, 'obs_only', False),
        action_only=getattr(cfg, 'action_only', False),
        normalizer=cfg.normalizer, particle_normalizer=cfg.particle_normalizer,
        preprocess_fns=cfg.preprocess_fns, use_padding=cfg.use_padding,
        max_path_length=cfg.max_path_length, overfit=False,
        single_view=(getattr(cfg, 'input_type', 'dlp') == "dlp" and not cfg.multiview),
        action_z_scale=getattr(cfg, 'action_z_scale', 1.0),
        use_gripper_obs=getattr(cfg, 'use_gripper_obs', False),
        use_bg_obs=getattr(cfg, 'use_bg_obs', False),
    )
    dataset = dataset_config()

    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim
    gripper_dim = getattr(dataset, 'gripper_dim', 0)
    bg_dim = getattr(dataset, 'bg_dim', 0)

    model_config = utils.Config(
        cfg.model, savepath=None, features_dim=cfg.features_dim,
        action_dim=action_dim, hidden_dim=cfg.hidden_dim,
        projection_dim=cfg.projection_dim, n_head=cfg.n_heads,
        n_layer=cfg.n_layers, dropout=cfg.dropout, block_size=cfg.horizon,
        positional_bias=cfg.positional_bias, max_particles=cfg.max_particles,
        multiview=cfg.multiview, device=cfg.device,
        gripper_dim=gripper_dim, bg_dim=bg_dim,
    )
    diffusion_config = utils.Config(
        cfg.diffusion, savepath=None, horizon=cfg.horizon,
        observation_dim=observation_dim, action_dim=action_dim,
        gripper_dim=gripper_dim, bg_dim=bg_dim,
        n_timesteps=cfg.n_diffusion_steps, loss_type=cfg.loss_type,
        clip_denoised=cfg.clip_denoised, predict_epsilon=cfg.predict_epsilon,
        action_weight=cfg.action_weight, loss_weights=cfg.loss_weights,
        loss_discount=cfg.loss_discount, device=cfg.device,
        obs_only=getattr(cfg, 'obs_only', False),
        action_only=getattr(cfg, 'action_only', False),
    )

    renderer = None
    model = model_config()
    diffusion = diffusion_config(model)
    trainer_config = utils.Config(
        utils.Trainer, savepath=None, train_batch_size=cfg.batch_size,
        train_lr=cfg.learning_rate,
        gradient_accumulate_every=cfg.gradient_accumulate_every,
        ema_decay=cfg.ema_decay, sample_freq=cfg.sample_freq,
        save_freq=cfg.save_freq, label_freq=int(cfg.n_train_steps // cfg.n_saves),
        save_parallel=cfg.save_parallel, results_folder=cfg.savepath,
        bucket=cfg.bucket, n_reference=cfg.n_reference,
    )
    trainer = trainer_config(diffusion, dataset, renderer)

    # Load checkpoint
    ckpt_data = torch.load(args.ckpt_path, map_location=cfg.device)
    trainer.step = ckpt_data['step']
    trainer.model.load_state_dict(ckpt_data['model'])
    trainer.ema_model.load_state_dict(ckpt_data['ema'])
    print(f"Loaded checkpoint at step {trainer.step}")

    # Setup env
    from diffuser.envs.mimicgen_dlp_wrapper import DatasetGoalProvider, MimicGenDLPWrapper
    from diffuser.eval_utils import extract_mimicgen_task_name, setup_mimicgen_env

    goal_provider = DatasetGoalProvider(dataset_path, shuffle=True)
    calib_h5_path = getattr(cfg, 'calib_h5_path')
    use_absolute_actions = getattr(cfg, 'use_absolute_actions', False)
    task = extract_mimicgen_task_name(calib_h5_path)
    exe_steps = getattr(cfg, 'exe_steps', 8)
    grid_dhw = getattr(cfg, 'mimicgen_grid_dhw', (128, 128, 128))
    cams = tuple(getattr(cfg, 'mimicgen_cams', ["agentview", "sideview"]))
    pixel_stride = getattr(cfg, 'mimicgen_pixel_stride', 1)

    multiview = getattr(cfg, 'multiview', False)

    # Override camera resolution to match preprocessing (84x84), NOT the config's 256x256.
    # The DLP was trained on 84x84 images (from preprocess_mimicgen_multiview.py),
    # resized to image_size=128 internally. Rendering at 256 then downsampling to 128
    # produces different image characteristics than rendering at 84 then upsampling to 128.
    cfg.mimicgen_camera_width = 84
    cfg.mimicgen_camera_height = 84

    print(f"Task: {task}, max_steps: {args.max_steps}, exe_steps: {exe_steps}, "
          f"is_3d: {is_3d}, multiview: {multiview}, cam_res: 84x84 (matching preprocessing)")

    device = torch.device(args.device)

    # Create env and wrapper once, reuse across episodes (like eval_paper.py)
    env = setup_mimicgen_env(cfg, use_absolute_actions=use_absolute_actions)
    envw = MimicGenDLPWrapper(
        env=env, dlp_model=dlp_model, device=args.device,
        cams=cams, grid_dhw=grid_dhw, pixel_stride=pixel_stride,
        calib_h5_path=calib_h5_path, goal_provider=goal_provider,
        random_init=args.random_init, normalize_to_unit_cube=False, task=task,
    )

    for ep in range(args.n_episodes):
        ep_seed = args.seed + ep
        print(f"\n{'='*50}")
        print(f"Episode {ep+1}/{args.n_episodes} (seed={ep_seed})")
        print(f"{'='*50}")

        ep_dir = os.path.join(args.output_dir, f"ep{ep:02d}_seed{ep_seed}")
        result = run_with_imagined_vis(
            trainer=trainer, dlp_model=dlp_model, envw=envw,
            dataset=dataset, device=device,
            max_steps=args.max_steps, exe_steps=exe_steps, seed=ep_seed,
            output_dir=ep_dir,
            save_horizon_frames=args.save_horizon_frames,
            max_replans_to_log=args.max_replans_to_log,
            is_3d=is_3d,
            multiview=multiview,
        )

        print(f"Episode {ep+1}: success={result['success']}, "
              f"length={result['length']}, replans={result['n_replans']}")

    # Close environment after all episodes
    try:
        env.close()
    except Exception:
        pass

    print(f"\nDone. Visualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
