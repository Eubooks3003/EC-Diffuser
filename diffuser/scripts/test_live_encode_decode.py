"""
Standalone test of the LIVE encoding + decoding pipeline.

For each listed demo episode:
  1. Launch RLBench
  2. reset_to_demo(demo)  -> sim state matches demo
  3. Replay the demo's actions one at a time
  4. At each step, grab the live RGB, encode via DLP, decode via DLP, save
  5. Write one mp4 per episode

This isolates whether the live encoder produces tokens that decode cleanly
(same DLP, same decode call as render_pkl_tokens.py, but the tokens come
from the LIVE sim instead of the offline pkl).

If these videos look as clean as render_pkl_tokens.py's output -> the live
encoder is fine, and any "noisy live_recon" you see at rollout time is
because the policy drives the sim to weird states the DLP can't cleanly
represent.

If these videos are still noisy -> the live encode path itself differs
from what preprocessing did (despite identical code), pointing to an
RGB-level difference or a DLP state issue.

Usage:
    cd /home/ellina/Desktop/EC-Diffuser
    export PYTHONPATH=/home/ellina/Desktop/lpwm-copy:$(pwd):$(pwd)/diffuser
    xvfb-run -a python diffuser/scripts/test_live_encode_decode.py \
        --episodes 0 1 2
"""
import argparse
import os
import pickle
import sys
import json

import numpy as np
import torch
import imageio.v2 as imageio


def build_dlp_2d(cfg, device, DLPClass):
    m = DLPClass(
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
        pint_enc_layers=cfg["pint_enc_layers"],
        pint_enc_heads=cfg["pint_enc_heads"],
        timestep_horizon=1,
    ).to(device)
    m.eval()
    return m


def _pack_tokens_2d(enc):
    z = enc["z"][:, 0]; z_scale = enc["z_scale"][:, 0]
    z_depth = enc["z_depth"][:, 0]; z_feat = enc["z_features"][:, 0]
    z_bg = enc["z_bg_features"][:, 0]
    obj_on = enc.get("z_obj_on", enc.get("obj_on"))[:, 0]
    if obj_on.dim() == 2:
        obj_on = obj_on.unsqueeze(-1)
    if z_bg.dim() == 3:
        z_bg = z_bg.squeeze(1)
    return torch.cat([z, z_scale, z_depth, obj_on, z_feat], dim=-1), z_bg


@torch.no_grad()
def encode_live_rgb(dlp_model, rgbs_uint8, image_size, device):
    """Mirror preprocessing: encode one view at a time (batch = frames of one
    view), NOT all views in a single batch. Preprocessing does per-view batching
    (see ec_diffuser_rlbench_multiview_preprocess.py:332-360) and any cross-
    batch dependency in the encoder makes per-view vs per-timestep batching
    produce different tokens for the same frame.
    """
    rgbs = torch.from_numpy(np.asarray(rgbs_uint8)).float() / 255.0
    rgbs = rgbs.permute(0, 3, 1, 2).to(device)  # (V, 3, H, W)
    if rgbs.shape[-1] != image_size:
        rgbs = torch.nn.functional.interpolate(
            rgbs, size=(image_size, image_size), mode="bilinear", align_corners=False
        )
    per_view_toks = []
    per_view_bg = []
    for vi in range(rgbs.shape[0]):
        chunk = rgbs[vi:vi + 1]  # (1, 3, H, W) — single view, single frame
        enc = dlp_model.encode_all(chunk, deterministic=True)
        toks, bg = _pack_tokens_2d(enc)  # (1, K, Dtok), (1, bg_F)
        per_view_toks.append(toks)
        per_view_bg.append(bg)
    toks = torch.cat(per_view_toks, dim=0)  # (V, K, Dtok)
    bg = torch.cat(per_view_bg, dim=0)      # (V, bg_F)
    return toks.cpu().numpy(), bg.cpu().numpy()


@torch.no_grad()
def decode_one_view(dlp_model, view_toks_np, view_bg_np, device):
    """Same decode as render_pkl_tokens.py and eval_rlbench_rollouts live_recon."""
    toks_t = torch.from_numpy(view_toks_np).float().unsqueeze(0).to(device)  # (1, K, D)
    z = toks_t[..., 0:2]
    z_scale = toks_t[..., 2:4]
    z_depth = toks_t[..., 4:5]
    obj_on = toks_t[..., 5:6]
    z_feat = toks_t[..., 6:]
    z_bg = torch.from_numpy(np.asarray(view_bg_np).flatten()).float().to(device).unsqueeze(0)
    dec = dlp_model.decode_all(z, z_scale, z_feat, obj_on, z_depth, z_bg, None, warmup=False)
    recon = dec["rec"].squeeze(0).permute(1, 2, 0).cpu().numpy()
    return (np.clip(recon, 0.0, 1.0) * 255).astype(np.uint8)


def _load_demo_from_disk(dataset_root, task_name, ep_idx):
    """Mirrors replay_demos.py. Loads from low_dim_obs.pkl only."""
    ep_dir = os.path.join(dataset_root, task_name, "all_variations", "episodes", f"episode{ep_idx}")
    with open(os.path.join(ep_dir, "low_dim_obs.pkl"), "rb") as f:
        demo = pickle.load(f)
    vp = os.path.join(ep_dir, "variation_number.pkl")
    if os.path.isfile(vp):
        with open(vp, "rb") as f:
            demo.variation_number = int(pickle.load(f))
    return demo


def rot6d_to_quat_xyzw(rot6d):
    """Inverse of quat_to_rot6d. Needed for replay action format."""
    rot6d = np.asarray(rot6d, dtype=np.float64).reshape(6)
    a1 = rot6d[:3]; a2 = rot6d[3:]
    b1 = a1 / (np.linalg.norm(a1) + 1e-12)
    a2_proj = a2 - np.dot(b1, a2) * b1
    b2 = a2_proj / (np.linalg.norm(a2_proj) + 1e-12)
    b3 = np.cross(b1, b2)
    R = np.stack([b1, b2, b3], axis=1)
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s; x = (m21 - m12) / s; y = (m02 - m20) / s; z = (m10 - m01) / s
    elif (m00 > m11) and (m00 > m22):
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s; x = 0.25 * s; y = (m01 + m10) / s; z = (m02 + m20) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s; x = (m01 + m10) / s; y = 0.25 * s; z = (m12 + m21) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s; x = (m02 + m20) / s; y = (m12 + m21) / s; z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    q /= (np.linalg.norm(q) + 1e-12)
    return q.astype(np.float32)


def stack_camera_rgbs(rlbench_obs, cams):
    rgbs = []
    for cam in cams:
        rgb = getattr(rlbench_obs, f"{cam}_rgb", None)
        if rgb is None:
            raise RuntimeError(f"missing {cam}_rgb on rlbench obs")
        rgbs.append(np.asarray(rgb))
    return np.stack(rgbs, axis=0).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="/home/ellina/Desktop/data/rlbench_preprocessed_multiview_tokens/rlbench_close_jar/rlbench_close_jar.pkl")
    ap.add_argument("--dlp-cfg", default="/home/ellina/Desktop/data/rlbench_preprocessed_multiview_tokens/rlbench_close_jar/dlp_config.json")
    ap.add_argument("--dlp-ckpt", default="/home/ellina/Desktop/data/rlbench_preprocessed_multiview_tokens/rlbench_close_jar/dlp_ckpt.pt")
    ap.add_argument("--dataset-root", default="/home/ellina/Desktop/data/rlbench_rgb",
                    help="Raw RLBench data root (contains <task>/all_variations/episodes/episodeN/low_dim_obs.pkl)")
    ap.add_argument("--task-name", default="close_jar")
    ap.add_argument("--lpwm-path", default="/home/ellina/Desktop/lpwm-copy")
    ap.add_argument("--out-dir", default="/home/ellina/Desktop/live_encode_decode_test")
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--cams", nargs="+", default=["front", "overhead"],
                    help="which camera views to use (matches eval's rlbench_cams)")
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--all-views", action="store_true")
    ap.add_argument("--max-steps", type=int, default=400)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sys.path.insert(0, args.lpwm_path)
    from models import DLP

    # --- Load pkl for ground-truth action sequences ---
    print(f"[test] loading pkl: {args.pkl}")
    with open(args.pkl, "rb") as f:
        data = pickle.load(f)
    actions_all = np.asarray(data["actions"])           # (E, Tmax, a_dim)
    path_lengths = np.asarray(data["path_lengths"])
    variations = np.asarray(data["variation_number"])
    print(f"[test] actions={actions_all.shape}")

    # --- Load DLP ---
    device = torch.device(args.device)
    with open(args.dlp_cfg) as f:
        cfg = json.load(f)
    model = build_dlp_2d(cfg, device, DLP)
    model.load_state_dict(torch.load(args.dlp_ckpt, map_location=device, weights_only=False))
    print(f"[test] DLP loaded  image_size={cfg['image_size']}  n_kp_enc={cfg['n_kp_enc']}")
    print(f"[test] model.training={model.training}")

    # (batch-size diagnostic moved inline to ep=0 t=0 below,
    # since we don't have local PNGs to test against)

    # --- Launch RLBench ---
    from rlbench import ObservationConfig, CameraConfig
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.environment import Environment
    from pyrep.const import RenderMode
    import importlib

    obs_config = ObservationConfig()
    obs_config.set_all(False)
    obs_config.gripper_pose = True
    obs_config.gripper_open = True
    cam_attrs = {
        "front": "front_camera",
        "overhead": "overhead_camera",
        "left_shoulder": "left_shoulder_camera",
        "right_shoulder": "right_shoulder_camera",
        "wrist": "wrist_camera",
    }
    for cam in args.cams:
        cc = CameraConfig(
            image_size=(args.image_size, args.image_size),
            rgb=True, depth=False, point_cloud=False, mask=False,
            render_mode=RenderMode.OPENGL,  # match demo collection; default is OPENGL3
        )
        setattr(obs_config, cam_attrs[cam], cc)

    action_mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaPlanning(),
        gripper_action_mode=Discrete(),
    )
    env = Environment(
        action_mode=action_mode,
        dataset_root=args.dataset_root,
        obs_config=obs_config,
        headless=True,
    )
    env.launch()

    task_module = importlib.import_module(f"rlbench.tasks.{args.task_name}")
    task_class = getattr(task_module, "".join(p.capitalize() for p in args.task_name.split("_")))
    task = env.get_task(task_class)

    V = len(args.cams)
    K_per_view = cfg["n_kp_enc"]
    view_names = list(args.cams)
    views_to_render = list(range(V)) if args.all_views else [0]

    try:
        for ep in args.episodes:
            if ep >= actions_all.shape[0]:
                continue
            T = int(path_lengths[ep])
            var = int(variations[ep])
            print(f"\n[test] ep={ep} var={var} T={T}")

            try:
                demo = _load_demo_from_disk(args.dataset_root, args.task_name, ep)
            except Exception as e:
                print(f"  load demo failed: {type(e).__name__}: {e}")
                continue
            demo_var = int(getattr(demo, "variation_number", var))
            task.set_variation(demo_var)
            _, rlbench_obs = task.reset_to_demo(demo)

            per_view_frames = {v: [] for v in views_to_render}
            raw_rgb_frames = {v: [] for v in views_to_render}

            # Load pkl tokens for this episode for direct numerical comparison
            pkl_obs = np.asarray(data["observations"])  # (E, Tmax, K_total, Dtok)
            pkl_bg = np.asarray(data["bg_features"])    # (E, Tmax, bg_total)
            K_total_pkl = pkl_obs.shape[2]
            K_per_view_pkl = int(cfg["n_kp_enc"])
            bg_per_view_pkl = pkl_bg.shape[2] // (K_total_pkl // K_per_view_pkl)

            # Initial frame (t=0, pre-action, matches demo frame 0)
            rgbs_np = stack_camera_rgbs(rlbench_obs, args.cams)  # (V, H, W, 3) uint8

            # Batch-size determinism check: encode the same t=0 front RGB in
            # batch=1 vs batch=32 and diff. Should be 0 if encoder is deterministic.
            if ep == args.episodes[0]:
                with torch.no_grad():
                    t1 = torch.from_numpy(rgbs_np[0]).float() / 255.0  # front view
                    t1 = t1.permute(2, 0, 1).unsqueeze(0).to(device)   # (1, 3, H, W)
                    enc1 = model.encode_all(t1, deterministic=True)
                    tok1, _ = _pack_tokens_2d(enc1)
                    t32 = t1.expand(32, -1, -1, -1).contiguous()
                    enc32 = model.encode_all(t32, deterministic=True)
                    tok32, _ = _pack_tokens_2d(enc32)
                    bdiff = (tok1[0] - tok32[0]).abs()
                    print(f"  [batch-size check] batch=1 vs batch=32 on same frame: "
                          f"tok_meanAbsDiff={bdiff.mean().item():.6f} "
                          f"maxAbsDiff={bdiff.max().item():.6f}")

            toks_v, bg_v = encode_live_rgb(model, rgbs_np, args.image_size, device)

            # Diff live tokens vs pkl tokens at t=0 (aligned: reset_to_demo + frame 0)
            print(f"  [token-diff t=0] — should be ~0 if encoder is consistent:")
            for v in views_to_render:
                live_t = toks_v[v]                                                # (K_per_view, D)
                pkl_t = pkl_obs[ep, 0, v * K_per_view_pkl:(v + 1) * K_per_view_pkl]  # (K_per_view, D)
                live_b = bg_v[v]                                                  # (bg_per_view,)
                pkl_b = pkl_bg[ep, 0, v * bg_per_view_pkl:(v + 1) * bg_per_view_pkl]
                mad_tok = np.mean(np.abs(live_t - pkl_t))
                max_tok = np.max(np.abs(live_t - pkl_t))
                mad_bg = np.mean(np.abs(live_b - pkl_b))
                print(f"    view={view_names[v]:>10s}: "
                      f"tok_meanAbsDiff={mad_tok:.5f} tok_maxAbsDiff={max_tok:.5f} "
                      f"bg_meanAbsDiff={mad_bg:.5f}")

            for v in views_to_render:
                per_view_frames[v].append(decode_one_view(model, toks_v[v], bg_v[v], device))
                raw_rgb_frames[v].append(rgbs_np[v])  # raw sim RGB at this step

            # Replay demo actions through sim, encoding + decoding at each step
            executed = 0
            for t_i in range(min(T, args.max_steps)):
                a = actions_all[ep, t_i]
                if np.allclose(a, 0.0):
                    break
                pos = a[:3].astype(np.float64)
                rot6d = a[3:9].astype(np.float64)
                gopen = 1.0 if float(a[9]) >= 0.5 else 0.0
                quat = rot6d_to_quat_xyzw(rot6d)
                # 9D: [pos(3), quat(4), gripper(1), ignore_collisions(1)]
                rlb_action = np.concatenate([pos.astype(np.float32), quat,
                                             [gopen], [1.0]]).astype(np.float32)
                try:
                    rlbench_obs, _, terminal = task.step(rlb_action)
                except Exception as e:
                    print(f"  step {t_i} failed: {type(e).__name__}: {str(e)[:120]}")
                    continue
                rgbs_np = stack_camera_rgbs(rlbench_obs, args.cams)
                toks_v, bg_v = encode_live_rgb(model, rgbs_np, args.image_size, device)
                for v in views_to_render:
                    per_view_frames[v].append(decode_one_view(model, toks_v[v], bg_v[v], device))
                    raw_rgb_frames[v].append(rgbs_np[v])
                executed += 1
                if terminal:
                    break

            for v in views_to_render:
                # Decoded reconstruction video (from live-encoded tokens)
                out_path = os.path.join(args.out_dir, f"ep{ep:02d}_var{var}_{view_names[v]}_live.mp4")
                imageio.mimsave(out_path, per_view_frames[v], fps=args.fps, macro_block_size=1)
                print(f"  wrote {out_path}  ({len(per_view_frames[v])} frames; {executed} demo actions replayed)")
                # Raw sim RGB video (what the encoder is being fed)
                raw_path = os.path.join(args.out_dir, f"ep{ep:02d}_var{var}_{view_names[v]}_raw.mp4")
                imageio.mimsave(raw_path, raw_rgb_frames[v], fps=args.fps, macro_block_size=1)
                print(f"  wrote {raw_path}  (raw sim RGB)")
    finally:
        env.shutdown()


if __name__ == "__main__":
    main()
