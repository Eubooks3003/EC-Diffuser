"""
Render one mp4 per demo from the preprocessed pkl's particle tokens.

For each episode listed via --episodes, decodes the front-view particle tokens
at every recorded timestep and writes an mp4. The slicing (first K_per_view
particles, first bg_per_view bg features = front view) matches exactly what
`eval_rlbench_rollouts` passes to the decoder in training.py's live_recon
path, so these videos are directly comparable to the eval's `*_live_recon.mp4`.

Usage:
    cd /home/ellina/Desktop/EC-Diffuser
    export PYTHONPATH=/home/ellina/Desktop/lpwm-copy:$(pwd):$(pwd)/diffuser
    python diffuser/scripts/render_pkl_tokens.py --episodes 0 1 2

Options:
    --all-views   also write per-view mp4s (overhead, left_shoulder, right_shoulder)
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


@torch.no_grad()
def decode_one_view(model, view_toks_np, view_bg_np, device):
    """Exactly matches the decode path used in eval_rlbench_rollouts' live_recon
    block (training.py:1048-1060) and origin/2D_DLP's
    visualize_imagined_states._decode_single_view_2d.

    Input slicing (10D particle token layout):
        z       = [..., 0:2]
        z_scale = [..., 2:4]
        z_depth = [..., 4:5]
        obj_on  = [..., 5:6]
        z_feat  = [..., 6:]
    """
    toks_t = torch.from_numpy(view_toks_np).float().unsqueeze(0).to(device)  # (1, K, D)
    z = toks_t[..., 0:2]
    z_scale = toks_t[..., 2:4]
    z_depth = toks_t[..., 4:5]
    obj_on = toks_t[..., 5:6]
    z_feat = toks_t[..., 6:]
    z_bg = torch.from_numpy(np.asarray(view_bg_np).flatten()).float().to(device).unsqueeze(0)
    dec = model.decode_all(z, z_scale, z_feat, obj_on, z_depth, z_bg, None, warmup=False)
    recon = dec["rec"].squeeze(0).permute(1, 2, 0).cpu().numpy()
    return (np.clip(recon, 0.0, 1.0) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="/home/ellina/Desktop/data/rlbench_preprocessed_multiview_tokens/rlbench_close_jar/rlbench_close_jar.pkl")
    ap.add_argument("--dlp-cfg", default="/home/ellina/Desktop/data/rlbench_preprocessed_multiview_tokens/rlbench_close_jar/dlp_config.json")
    ap.add_argument("--dlp-ckpt", default="/home/ellina/Desktop/data/rlbench_preprocessed_multiview_tokens/rlbench_close_jar/dlp_ckpt.pt")
    ap.add_argument("--lpwm-path", default="/home/ellina/Desktop/lpwm-copy")
    ap.add_argument("--out-dir", default="/home/ellina/Desktop/pkl_render_test")
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--all-views", action="store_true",
                    help="also write mp4s for overhead/left_shoulder/right_shoulder")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    sys.path.insert(0, args.lpwm_path)
    from models import DLP

    print(f"[render] loading {args.pkl}")
    with open(args.pkl, "rb") as f:
        data = pickle.load(f)
    obs = np.asarray(data["observations"])           # (E, Tmax, K_total, Dtok)
    bg = np.asarray(data["bg_features"])             # (E, Tmax, bg_total)
    path_lengths = np.asarray(data["path_lengths"])
    langs = data.get("language")
    variations = np.asarray(data.get("variation_number", np.zeros(obs.shape[0], dtype=int)))
    print(f"[render] obs={obs.shape} bg={bg.shape}")

    device = torch.device(args.device)
    with open(args.dlp_cfg) as f:
        cfg = json.load(f)
    print(f"[render] building DLP (n_kp_enc={cfg['n_kp_enc']}, image_size={cfg['image_size']})")
    model = build_dlp_2d(cfg, device, DLP)
    model.load_state_dict(torch.load(args.dlp_ckpt, map_location=device, weights_only=False))

    K_total = obs.shape[2]
    bg_total = bg.shape[2]
    K_per_view = int(cfg["n_kp_enc"])
    V = K_total // K_per_view
    bg_per_view = bg_total // V
    print(f"[render] V={V} K_per_view={K_per_view} bg_per_view={bg_per_view}")
    view_names = ["front", "overhead", "left_shoulder", "right_shoulder"][:V]

    # Which views to render: front only by default; all if --all-views.
    if args.all_views:
        views_to_render = list(range(V))
    else:
        views_to_render = [0]  # front only — matches live recon

    for ep in args.episodes:
        if ep >= obs.shape[0]:
            print(f"[render] skipping ep={ep} (pkl has only {obs.shape[0]} demos)")
            continue
        T = int(path_lengths[ep])
        lang = ""
        if langs is not None and ep < len(langs):
            l = langs[ep]
            lang = l[0] if isinstance(l, (list, tuple)) and l else str(l)
        var = int(variations[ep])
        print(f"[render] ep={ep} var={var} T={T} lang={lang!r}")

        frames_per_view = {v: [] for v in views_to_render}
        for fr in range(T):
            for v in views_to_render:
                vt = obs[ep, fr, v * K_per_view:(v + 1) * K_per_view]
                vbg = bg[ep, fr, v * bg_per_view:(v + 1) * bg_per_view]
                frames_per_view[v].append(decode_one_view(model, vt, vbg, device))

        for v in views_to_render:
            out_path = os.path.join(args.out_dir,
                                    f"ep{ep:02d}_var{var}_{view_names[v]}.mp4")
            imageio.mimsave(out_path, frames_per_view[v], fps=args.fps, macro_block_size=1)
            print(f"  wrote {out_path}  ({len(frames_per_view[v])} frames)")


if __name__ == "__main__":
    main()
