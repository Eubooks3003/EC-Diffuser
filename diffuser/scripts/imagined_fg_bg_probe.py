#!/usr/bin/env python
"""
Render imagined FG / BG / composite separately, and report per-channel drift.

Two questions this answers that the combined render cannot:

  1. Fewer active particles explains an EMPTIER scene, not a WRONG one. By
     decoding the foreground (particle) branch and the background branch
     separately we can see which one produces the hallucinated content.
     bg_features is only 4 dims per view but drives the ENTIRE background
     image, so a small drift there rewrites the whole scene appearance --
     which would look like "completely wrong imagined features" even if every
     particle were perfect.

  2. Which token channels actually drift. Reported per channel group, over
     ACTIVE particles only (obj_on > 0.5), because inactive slots carry
     arbitrary values and would swamp the average.

Runs on dataset trajectories -- no simulator. Ground truth for the same
horizon steps is decoded alongside, so each panel has a reference.

Usage:
    PYTHONPATH=.:.. python scripts/imagined_fg_bg_probe.py \
        --config mimicgen224_multitask_semantic_dlp --mode 12C_dlp \
        --ckpt_path <ckpt.pt> --eval_task stack --out_dir /tmp/fgbg
"""
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..")),
          os.path.abspath(os.path.join(SCRIPT_DIR, ".."))):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import diffuser.utils as utils
# Reuse the visualizer's DLP loader (it carries the lpwm-copy/lpwm-dev sys.path
# preference fix); the config/model setup is inlined below so this script does
# not depend on the visualizer's internals.
from visualize_imagined_states import load_dlp_lpwm  # noqa: E402


@torch.no_grad()
def decode_parts(dlp, toks_np, bg_np, device):
    """Decode one view -> dict of uint8 images: composite / bg / fg-only."""
    t = torch.from_numpy(toks_np).float().unsqueeze(0).to(device)
    z, z_scale, z_depth = t[..., 0:2], t[..., 2:4], t[..., 4:5]
    obj_on, z_feat = t[..., 5:6], t[..., 6:]
    z_bg = None
    if bg_np is not None:
        z_bg = torch.from_numpy(np.asarray(bg_np).flatten()).float().to(device)[None]
    dec = dlp.decode_all(z, z_scale, z_feat, obj_on, z_depth, z_bg, None, warmup=False)

    def img(x):
        a = x.squeeze(0).detach().cpu().numpy()
        if a.ndim == 4:            # (K,C,H,W) glimpses -> max over particles
            a = a.max(axis=0)
        a = a.transpose(1, 2, 0)
        return (np.clip(a, 0, 1) * 255).astype(np.uint8)

    out = {"composite": img(dec["rec_rgb"]), "bg": img(dec["bg_rgb"])}
    if "dec_objects_trans" in dec:
        out["fg"] = img(dec["dec_objects_trans"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", default="12C_dlp")
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--eval_task", default=None)
    ap.add_argument("--sample_idx", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out_dir", default="/tmp/fgbg")
    ap.add_argument("--h_steps", type=str, default="0,1,2,4,8,15")
    args = ap.parse_args()

    import importlib
    cfg_mod = importlib.import_module(f"config.{args.config}")
    merged = {**cfg_mod.base["diffusion"], **cfg_mod.mode_to_args.get(args.mode, {})}

    class Cfg:
        pass
    cfg = Cfg()
    for k, v in merged.items():
        setattr(cfg, k, v)
    cfg.device = args.device
    device = torch.device(args.device)

    if getattr(cfg, "multitask", False):
        if not args.eval_task:
            raise RuntimeError("multitask config: --eval_task required")
        m = next((e for e in (getattr(cfg, "task_entries", []) or [])
                  if e["name"] == args.eval_task), None)
        if m is None:
            raise RuntimeError(f"unknown --eval_task {args.eval_task}")
        cfg.override_dataset_path = m["pkl"]
        cfg.dlp_ckpt, cfg.dlp_cfg = m["dlp_ckpt"], m["dlp_cfg"]

    dataset = utils.Config(
        cfg.loader, savepath=None,
        dataset_path=getattr(cfg, "override_dataset_path", None),
        dataset_name=cfg.dataset, horizon=cfg.horizon,
        obs_only=getattr(cfg, "obs_only", False),
        action_only=getattr(cfg, "action_only", False),
        normalizer=cfg.normalizer, particle_normalizer=cfg.particle_normalizer,
        preprocess_fns=cfg.preprocess_fns, use_padding=cfg.use_padding,
        max_path_length=cfg.max_path_length, overfit=False,
        single_view=(getattr(cfg, "input_type", "dlp") == "dlp" and not cfg.multiview),
        action_z_scale=getattr(cfg, "action_z_scale", 1.0),
        use_gripper_obs=getattr(cfg, "use_gripper_obs", False),
        use_bg_obs=getattr(cfg, "use_bg_obs", False),
        task_entries=getattr(cfg, "task_entries", None),
        max_demos_per_task=getattr(cfg, "max_demos_per_task", None),
    )()

    model = utils.Config(
        cfg.model, savepath=None, features_dim=cfg.features_dim,
        action_dim=dataset.action_dim, hidden_dim=cfg.hidden_dim,
        projection_dim=cfg.projection_dim, n_head=cfg.n_heads,
        n_layer=cfg.n_layers, dropout=cfg.dropout, block_size=cfg.horizon,
        positional_bias=cfg.positional_bias, max_particles=cfg.max_particles,
        multiview=cfg.multiview, device=cfg.device,
        gripper_dim=getattr(dataset, "gripper_dim", 0),
        bg_dim=getattr(dataset, "bg_dim", 0),
        n_tasks=getattr(cfg, "n_tasks", 1),
        split_action_tokens=getattr(cfg, "split_action_tokens", None),
        action_token_groups=getattr(cfg, "action_token_groups", None),
        proprio_token_groups=getattr(cfg, "proprio_token_groups", None),
        aux_action_token_groups=getattr(cfg, "aux_action_token_groups", None),
        aux_proprio_token_groups=getattr(cfg, "aux_proprio_token_groups", None),
    )()
    diffusion = utils.Config(
        cfg.diffusion, savepath=None, horizon=cfg.horizon,
        observation_dim=dataset.observation_dim, action_dim=dataset.action_dim,
        gripper_dim=getattr(dataset, "gripper_dim", 0),
        bg_dim=getattr(dataset, "bg_dim", 0),
        n_timesteps=cfg.n_diffusion_steps, loss_type=cfg.loss_type,
        clip_denoised=cfg.clip_denoised, predict_epsilon=cfg.predict_epsilon,
        action_weight=cfg.action_weight, loss_weights=cfg.loss_weights,
        loss_discount=cfg.loss_discount, device=cfg.device,
        obs_only=getattr(cfg, "obs_only", False),
        action_only=getattr(cfg, "action_only", False),
    )(model)
    ck = torch.load(args.ckpt_path, map_location=cfg.device, weights_only=False)
    diffusion.load_state_dict(ck.get("ema", ck.get("model")))
    diffusion.to(device).eval()

    dlp, _ = load_dlp_lpwm(cfg.dlp_cfg, cfg.dlp_ckpt, cfg.device,
                           getattr(cfg, "dlp_ctor", "models:DLP"))
    H = cfg.horizon
    a_dim, g_dim = dataset.action_dim, getattr(dataset, "gripper_dim", 0)
    bg_dim = getattr(dataset, "bg_dim", 0)
    obs_start = a_dim + g_dim + bg_dim
    bg_s, bg_e = a_dim + g_dim, a_dim + g_dim + bg_dim
    n_views = 2 if cfg.multiview else 1
    F = cfg.features_dim

    # A multitask dataset pools every task, so dataset[0] is whatever task sorts
    # first -- NOT the one named by --eval_task. Selecting the DLP/pkl for a task
    # while drawing a sample from another silently renders the wrong scene under
    # the right label. Restrict to indices whose episode carries this task_id.
    idx = args.sample_idx if args.sample_idx is not None else 0
    if getattr(cfg, "multitask", False):
        want = int(m["task_id"])
        tid = dataset.fields.task_ids
        cand = [i for i, (pi, _s, _e) in enumerate(dataset.indices)
                if int(tid[pi]) == want]
        if not cand:
            raise RuntimeError(f"no dataset samples for task_id={want} ({args.eval_task})")
        idx = cand[idx % len(cand)]
        print(f"[sample] task={args.eval_task} task_id={want} -> dataset index {idx} "
              f"({len(cand)} candidates)")
    batch = dataset[idx]
    traj = np.asarray(batch.trajectories)
    cond = {k: torch.as_tensor(v, dtype=torch.float32, device=device)[None]
            for k, v in batch.conditions.items()}
    kw = {}
    if hasattr(batch, "task_id"):
        kw["task_id"] = torch.as_tensor([int(batch.task_id)], device=device)
    with torch.no_grad():
        pred = diffusion(cond, verbose=False, **kw)
    pred = np.asarray((pred.trajectories if hasattr(pred, "trajectories") else pred)[0].cpu())

    p_obs = dataset.normalizer.unnormalize(pred[:, obs_start:], "observations")
    t_obs = dataset.normalizer.unnormalize(traj[:, obs_start:], "observations")
    p_bg = dataset.normalizer.unnormalize(pred[:, bg_s:bg_e], "bg_features") if bg_dim else None
    t_bg = dataset.normalizer.unnormalize(traj[:, bg_s:bg_e], "bg_features") if bg_dim else None
    K = p_obs.shape[-1] // F

    # ---- per-channel drift, active particles only -------------------------
    groups = {"z(pos)": (0, 2), "z_scale": (2, 4), "z_depth": (4, 5),
              "obj_on": (5, 6), "z_features": (6, F)}
    print(f"\nPer-channel drift  |pred - true|, ACTIVE particles only (obj_on>0.5)")
    print("  h  | " + " ".join(f"{g:>11s}" for g in groups) + " |    bg |  n_act")
    print("-" * 92)
    for h in range(H):
        pt = p_obs[h].reshape(K, F)
        tt = t_obs[h].reshape(K, F)
        m = tt[:, 5] > 0.5
        row = []
        for g, (s, e) in groups.items():
            row.append(np.abs(pt[m, s:e] - tt[m, s:e]).mean() if m.any() else np.nan)
        bg_d = np.abs(p_bg[h] - t_bg[h]).mean() if bg_dim else np.nan
        print(f" {h:3d} | " + " ".join(f"{v:11.4f}" for v in row) +
              f" | {bg_d:5.3f} | {int(m.sum()):3d}")

    if bg_dim:
        print(f"\nbg_features true range over horizon: "
              f"[{t_bg.min():.3f}, {t_bg.max():.3f}]  "
              f"drift h0->h15: {np.abs(p_bg[0]-t_bg[0]).mean():.4f} -> "
              f"{np.abs(p_bg[-1]-t_bg[-1]).mean():.4f}")

    # ---- render fg / bg / composite ---------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    hs = [int(x) for x in args.h_steps.split(",") if int(x) < H]
    Kv, bgv = K // n_views, (bg_dim // n_views if bg_dim else 0)

    for v in range(n_views):
        rows = ["composite", "bg", "fg"]
        fig, axes = plt.subplots(len(rows) * 2, len(hs),
                                 figsize=(2.1 * len(hs), 2.1 * len(rows) * 2))
        for j, h in enumerate(hs):
            for src, off in (("IMAG", 0), ("TRUE", len(rows))):
                o = p_obs if src == "IMAG" else t_obs
                b = (p_bg if src == "IMAG" else t_bg)
                tk = o[h].reshape(K, F)[v * Kv:(v + 1) * Kv]
                bb = b[h][v * bgv:(v + 1) * bgv] if bg_dim else None
                parts = decode_parts(dlp, tk, bb, device)
                for i, r in enumerate(rows):
                    ax = axes[off + i, j]
                    ax.imshow(parts.get(r, np.zeros_like(parts["composite"])))
                    ax.axis("off")
                    if j == 0:
                        ax.set_ylabel(f"{src}\n{r}", fontsize=8)
                        ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
                    if off == 0 and i == 0:
                        ax.set_title(f"h={h}", fontsize=9)
        fig.suptitle(f"view {v}: imagined vs true, split into background / foreground",
                     fontsize=11)
        fig.tight_layout()
        p = os.path.join(args.out_dir, f"fg_bg_view{v}.png")
        fig.savefig(p, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
