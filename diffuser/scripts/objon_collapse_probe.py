#!/usr/bin/env python
"""
Why does predicted obj_on collapse over the horizon?

Observed: the model keeps the right particles on at h=0 (pinned by conditioning)
and progressively switches them off as h grows, dropping the gripper and the
manipulated object by h~8.

The leading hypothesis is that L1's minimiser is the conditional MEDIAN, which
for a near-binary variable is a threshold rule: predict on iff
P(on | conditioning) > 0.5. Once conditioning stops being informative, that
probability falls to each slot's marginal rate -- and if occupancy is under 50%,
the loss-optimal prediction is "off" for every slot.

That hypothesis makes two falsifiable predictions this script tests:

  1. Tasks with HIGHER true occupancy should retain more particles. If retention
     is flat across occupancy, the median story is wrong.
  2. The slots the model keeps on should be the ones with the HIGHEST marginal
     activity rate -- it should drop the rarely-active slots first, not random
     ones.

All 12 tasks share one pooled dataset, so it is loaded once and indices are
filtered per task.

Usage:
    PYTHONPATH=.:.. python scripts/objon_collapse_probe.py \
        --config mimicgen224_multitask_semantic_dlp --ckpt_path <ckpt.pt> \
        --n_samples 12
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
for p in (os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..")),
          os.path.abspath(os.path.join(SCRIPT_DIR, ".."))):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import diffuser.utils as utils


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", default="12C_dlp")
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--n_samples", type=int, default=12)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import importlib
    cm = importlib.import_module(f"config.{args.config}")
    merged = {**cm.base["diffusion"], **cm.mode_to_args.get(args.mode, {})}

    class Cfg:
        pass
    cfg = Cfg()
    for k, v in merged.items():
        setattr(cfg, k, v)
    cfg.device = args.device
    dev = torch.device(args.device)

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

    a_dim = dataset.action_dim
    g_dim = getattr(dataset, "gripper_dim", 0)
    bg_dim = getattr(dataset, "bg_dim", 0)
    obs_start = a_dim + g_dim + bg_dim
    F = cfg.features_dim
    H = cfg.horizon

    model = utils.Config(
        cfg.model, savepath=None, features_dim=F, action_dim=a_dim,
        hidden_dim=cfg.hidden_dim, projection_dim=cfg.projection_dim,
        n_head=cfg.n_heads, n_layer=cfg.n_layers, dropout=cfg.dropout,
        block_size=H, positional_bias=cfg.positional_bias,
        max_particles=cfg.max_particles, multiview=cfg.multiview,
        device=cfg.device, gripper_dim=g_dim, bg_dim=bg_dim,
        n_tasks=getattr(cfg, "n_tasks", 1),
        split_action_tokens=getattr(cfg, "split_action_tokens", None),
        action_token_groups=getattr(cfg, "action_token_groups", None),
        proprio_token_groups=getattr(cfg, "proprio_token_groups", None),
        aux_action_token_groups=getattr(cfg, "aux_action_token_groups", None),
        aux_proprio_token_groups=getattr(cfg, "aux_proprio_token_groups", None),
    )()
    diffusion = utils.Config(
        cfg.diffusion, savepath=None, horizon=H,
        observation_dim=dataset.observation_dim, action_dim=a_dim,
        gripper_dim=g_dim, bg_dim=bg_dim, n_timesteps=cfg.n_diffusion_steps,
        loss_type=cfg.loss_type, clip_denoised=cfg.clip_denoised,
        predict_epsilon=cfg.predict_epsilon, action_weight=cfg.action_weight,
        loss_weights=cfg.loss_weights, loss_discount=cfg.loss_discount,
        device=cfg.device, obs_only=getattr(cfg, "obs_only", False),
        action_only=getattr(cfg, "action_only", False),
    )(model)
    ck = torch.load(args.ckpt_path, map_location=cfg.device, weights_only=False)
    diffusion.load_state_dict(ck.get("ema", ck.get("model")))
    diffusion.to(dev).eval()

    entries = {e["name"]: int(e["task_id"]) for e in cfg.task_entries}
    tid_field = dataset.fields.task_ids
    by_task = {}
    for i, (pi, _s, _e) in enumerate(dataset.indices):
        by_task.setdefault(int(tid_field[pi]), []).append(i)

    rng = np.random.RandomState(args.seed)
    rows = []
    # slot-level: marginal activity rate vs whether the model keeps it on
    slot_keep, slot_rate = [], []

    for name, tid in sorted(entries.items(), key=lambda kv: kv[1]):
        cand = by_task.get(tid, [])
        if not cand:
            continue
        idxs = rng.choice(cand, size=min(args.n_samples, len(cand)), replace=False)
        t_act, p_act = [], []
        marg = np.zeros(0)
        for j in idxs:
            b = dataset[int(j)]
            traj = np.asarray(b.trajectories)
            cond = {k: torch.as_tensor(v, dtype=torch.float32, device=dev)[None]
                    for k, v in b.conditions.items()}
            kw = {"task_id": torch.as_tensor([int(b.task_id)], device=dev)} \
                if hasattr(b, "task_id") else {}
            with torch.no_grad():
                pr = diffusion(cond, verbose=False, **kw)
            pr = np.asarray((pr.trajectories if hasattr(pr, "trajectories") else pr)[0].cpu())
            po = dataset.normalizer.unnormalize(pr[:, obs_start:], "observations")
            to = dataset.normalizer.unnormalize(traj[:, obs_start:], "observations")
            K = po.shape[-1] // F
            pon = po.reshape(H, K, F)[..., 5] > 0.5
            ton = to.reshape(H, K, F)[..., 5] > 0.5
            t_act.append(ton.sum(1))
            p_act.append(pon.sum(1))
            if marg.size == 0:
                marg = np.zeros(K)
            marg += ton.mean(0)
            # late-horizon keep decision per slot, against that slot's true rate
            slot_keep.append(pon[H // 2:].mean(0))
            slot_rate.append(ton.mean(0))

        t_act = np.mean(t_act, 0)
        p_act = np.mean(p_act, 0)
        occ = t_act.mean() / K
        ret = p_act[-1] / max(p_act[0], 1e-9)
        rows.append((name, K, occ, p_act[0], p_act[-1], ret))

    print(f"\n{'task':24s} {'occupancy':>10s} {'pred act h0':>12s} {'h15':>7s} {'retained':>9s}")
    print("-" * 68)
    for n, K, occ, a0, a15, ret in rows:
        print(f"{n:24s} {occ*100:9.1f}% {a0:12.1f} {a15:7.1f} {ret*100:8.1f}%")

    occ = np.array([r[2] for r in rows])
    ret = np.array([r[5] for r in rows])
    if len(occ) > 2:
        c = np.corrcoef(occ, ret)[0, 1]
        print(f"\nPrediction 1 -- occupancy vs retention: corr = {c:+.3f}")
        print("  (median hypothesis predicts a clear POSITIVE correlation)")

    sk = np.concatenate(slot_keep)
    sr = np.concatenate(slot_rate)
    if sk.size:
        kept = sr[sk > 0.5]
        dropped = sr[sk <= 0.5]
        print(f"\nPrediction 2 -- true activity rate of slots the model KEEPS vs DROPS:")
        print(f"  kept   : n={kept.size:6d}  mean true rate={kept.mean():.3f}")
        print(f"  dropped: n={dropped.size:6d}  mean true rate={dropped.mean():.3f}")
        print("  (median hypothesis predicts kept >> dropped)")

    if args.out:
        json.dump({"rows": rows, "corr": float(c) if len(occ) > 2 else None},
                  open(args.out, "w"), indent=2, default=float)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
