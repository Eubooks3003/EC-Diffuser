#!/usr/bin/env python
"""
Measure how the diffuser's prediction degrades with horizon, offline.

Answers one question: does the ACTION chunk degrade at the same horizon step
the imagined PARTICLES do, or does it stay usable while the particles fall
apart? The first case means the shared trunk fails past some h and the action
chunking is not delivering the steps it was designed for; the second means the
imagined observations are cosmetic and low task success lies elsewhere.

Runs on dataset trajectories only -- no simulator, no rollouts.

For each sampled trajectory the model is conditioned exactly as it is at
rollout (current observation at h=0) and produces the full H-step plan. We then
compare against ground truth at every h:

  action error   : mean |pred - true| over the action dims
  particle error : symmetric Chamfer distance between predicted and true
                   particle SETS -- permutation-invariant, because DLP slot
                   identity is not stable across frames, so a slot-wise
                   comparison would report differences that are only reorderings
  per view       : entities are split per camera before matching (a particle in
                   view 0 must not match one in view 1)

Both are reported against a PERSISTENCE baseline -- repeat a0 / copy the h=0
particles. Raw error values are not interpretable on their own; what matters is
the horizon at which the model stops beating "assume nothing changes".

Usage:
    PYTHONPATH=.:.. python scripts/horizon_error_probe.py \
        --config mimicgen224_multitask_semantic_dlp --mode 12C_dlp \
        --ckpt_path <ckpt.pt> --eval_task stack --n_samples 64 \
        --out /tmp/horizon_probe.json
"""
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import json
import argparse
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DIFFUSER_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
EC_DIFFUSER_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
for p in (EC_DIFFUSER_ROOT, DIFFUSER_ROOT):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import diffuser.utils as utils


def chamfer(a, b):
    """Symmetric Chamfer distance between two point sets. a,b: (N,D), (M,D)."""
    if a.shape[0] == 0 or b.shape[0] == 0:
        return float("nan")
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)   # (N,M)
    return float(d.min(axis=1).mean() + d.min(axis=0).mean()) / 2.0


def particle_err(pred_obs, true_obs, n_views, feat_dim, pos_dims=2):
    """Chamfer over full tokens and over position dims only, per view."""
    pred = pred_obs.reshape(-1, feat_dim)
    true = true_obs.reshape(-1, feat_dim)
    per_view = max(1, pred.shape[0] // n_views)
    full, pos = [], []
    for v in range(n_views):
        p = pred[v * per_view:(v + 1) * per_view]
        t = true[v * per_view:(v + 1) * per_view]
        full.append(chamfer(p, t))
        pos.append(chamfer(p[:, :pos_dims], t[:, :pos_dims]))
    return float(np.mean(full)), float(np.mean(pos))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", default="12C_dlp")
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--eval_task", default=None)
    ap.add_argument("--n_samples", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
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

    if getattr(cfg, "multitask", False):
        if not args.eval_task:
            raise RuntimeError("multitask config: --eval_task required")
        entries = getattr(cfg, "task_entries", []) or []
        m = next((e for e in entries if e["name"] == args.eval_task), None)
        if m is None:
            raise RuntimeError(f"task {args.eval_task} not in {[e['name'] for e in entries]}")
        cfg.override_dataset_path = m["pkl"]

    dataset_config = utils.Config(
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
    )
    dataset = dataset_config()

    obs_dim = dataset.observation_dim
    act_dim = dataset.action_dim
    grip_dim = getattr(dataset, "gripper_dim", 0)
    bg_dim = getattr(dataset, "bg_dim", 0)

    model_config = utils.Config(
        cfg.model, savepath=None, features_dim=cfg.features_dim,
        action_dim=act_dim, hidden_dim=cfg.hidden_dim,
        projection_dim=cfg.projection_dim, n_head=cfg.n_heads,
        n_layer=cfg.n_layers, dropout=cfg.dropout, block_size=cfg.horizon,
        positional_bias=cfg.positional_bias, max_particles=cfg.max_particles,
        multiview=cfg.multiview, device=cfg.device,
        gripper_dim=grip_dim, bg_dim=bg_dim,
        n_tasks=getattr(cfg, "n_tasks", 1),
        split_action_tokens=getattr(cfg, "split_action_tokens", None),
        action_token_groups=getattr(cfg, "action_token_groups", None),
        proprio_token_groups=getattr(cfg, "proprio_token_groups", None),
        aux_action_token_groups=getattr(cfg, "aux_action_token_groups", None),
        aux_proprio_token_groups=getattr(cfg, "aux_proprio_token_groups", None),
    )
    diffusion_config = utils.Config(
        cfg.diffusion, savepath=None, horizon=cfg.horizon,
        observation_dim=obs_dim, action_dim=act_dim,
        gripper_dim=grip_dim, bg_dim=bg_dim,
        n_timesteps=cfg.n_diffusion_steps, loss_type=cfg.loss_type,
        clip_denoised=cfg.clip_denoised, predict_epsilon=cfg.predict_epsilon,
        action_weight=cfg.action_weight, loss_weights=cfg.loss_weights,
        loss_discount=cfg.loss_discount, device=cfg.device,
        obs_only=getattr(cfg, "obs_only", False),
        action_only=getattr(cfg, "action_only", False),
    )
    model = model_config()
    diffusion = diffusion_config(model)

    ckpt = torch.load(args.ckpt_path, map_location=cfg.device, weights_only=False)
    state = ckpt.get("ema", ckpt.get("model"))
    diffusion.load_state_dict(state)
    diffusion.to(cfg.device).eval()
    print(f"[probe] loaded {os.path.basename(args.ckpt_path)}  "
          f"H={cfg.horizon} obs_dim={obs_dim} act_dim={act_dim} "
          f"grip={grip_dim} bg={bg_dim}")

    n_views = 2 if cfg.multiview else 1
    feat_dim = cfg.features_dim
    H = cfg.horizon
    rng = np.random.RandomState(args.seed)
    idxs = rng.choice(len(dataset), size=min(args.n_samples, len(dataset)), replace=False)

    acc = {k: np.zeros((H,)) for k in
           ("act", "act_base", "part_full", "part_pos", "part_full_base", "part_pos_base")}
    n = 0
    obs_start = act_dim + grip_dim + bg_dim

    for i, idx in enumerate(idxs):
        batch = dataset[int(idx)]
        traj = np.asarray(batch.trajectories)
        cond = {k: torch.as_tensor(v, dtype=torch.float32, device=cfg.device)[None]
                for k, v in batch.conditions.items()}
        tid = None
        if hasattr(batch, "task_id"):
            tid = torch.as_tensor([int(batch.task_id)], device=cfg.device)

        with torch.no_grad():
            pred = diffusion(cond, verbose=False, **({"task_id": tid} if tid is not None else {}))
        pred = np.asarray(pred[0].detach().cpu()) if not hasattr(pred, "trajectories") \
            else np.asarray(pred.trajectories[0].detach().cpu())

        for h in range(H):
            acc["act"][h] += np.abs(pred[h, :act_dim] - traj[h, :act_dim]).mean()
            # persistence: repeat the true first action
            acc["act_base"][h] += np.abs(traj[0, :act_dim] - traj[h, :act_dim]).mean()

            pf, pp = particle_err(pred[h, obs_start:], traj[h, obs_start:], n_views, feat_dim)
            acc["part_full"][h] += pf
            acc["part_pos"][h] += pp
            # persistence: copy the true particles at h=0
            bf, bp = particle_err(traj[0, obs_start:], traj[h, obs_start:], n_views, feat_dim)
            acc["part_full_base"][h] += bf
            acc["part_pos_base"][h] += bp
        n += 1
        if (i + 1) % 16 == 0:
            print(f"  {i+1}/{len(idxs)}")

    for k in acc:
        acc[k] /= max(n, 1)

    print(f"\n{'h':>3} | {'action':>9} {'a-persist':>10} | {'part(full)':>11} "
          f"{'p-persist':>10} | {'part(xy)':>9} {'xy-persist':>11}")
    print("-" * 76)
    for h in range(H):
        print(f"{h:>3} | {acc['act'][h]:9.4f} {acc['act_base'][h]:10.4f} | "
              f"{acc['part_full'][h]:11.4f} {acc['part_full_base'][h]:10.4f} | "
              f"{acc['part_pos'][h]:9.4f} {acc['part_pos_base'][h]:11.4f}")

    out = {"config": args.config, "ckpt": args.ckpt_path, "eval_task": args.eval_task,
           "horizon": H, "n_samples": n, "obs_dim": obs_dim, "action_dim": act_dim,
           **{k: v.tolist() for k, v in acc.items()}}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
