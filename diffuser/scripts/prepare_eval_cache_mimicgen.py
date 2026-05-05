#!/usr/bin/env python
"""
One-time builder for the multitask paper-eval dataset cache.

Builds the MultitaskGoalDataset (~10 GB peak: loads all 12 task pkls, fits a
GaussianNormalizer over valid frames across tasks), then pickles only the
fields the eval workers actually read: normalizer, *_dim, particle_dim,
horizon, action_z_scale. Each eval_paper.py worker then loads this cache
(<10 MB) instead of re-fitting the normalizer from raw pkls. With
concurrency=12 this drops worker resident-RAM from ~10 GB to ~3-5 GB and
keeps total RAM under the box's physical capacity.

Output: <savepath>/eval_cache.pkl
        savepath = os.path.dirname(ckpt_path).replace('/ckpt', '')
        i.e. the same dir as ckpt/, dataset_config.pkl, etc.

Usage:
    python diffuser/scripts/prepare_eval_cache_mimicgen.py \\
        --config mimicgen_multitask_dlp \\
        --mode 12C_dlp \\
        --ckpt_path data/multitask/diffusion/.../ckpt/state_*.pt

Flags:
    --force      Rebuild the cache even if <savepath>/eval_cache.pkl exists.
"""
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import argparse
import pickle
import time
from datetime import datetime

# path setup (same as eval_paper.py)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DIFFUSER_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
EC_DIFFUSER_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
for p in (EC_DIFFUSER_ROOT, DIFFUSER_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

CACHE_FILENAME = "eval_cache.pkl"
CACHE_VERSION  = 1


def main():
    parser = argparse.ArgumentParser(description="Prep multitask eval cache (normalizer + dims).")
    parser.add_argument("--config", type=str, required=True,
                        help="Config module under diffuser/config (e.g. mimicgen_multitask_dlp)")
    parser.add_argument("--mode", type=str, required=True,
                        help="Mode key in config.mode_to_args (e.g. 12C_dlp)")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to the multitask checkpoint .pt (used to derive savepath)")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if <savepath>/eval_cache.pkl already exists")
    args = parser.parse_args()

    savepath = os.path.dirname(os.path.abspath(args.ckpt_path)).replace("/ckpt", "")
    cache_path = os.path.join(savepath, CACHE_FILENAME)
    print(f"[prepare_eval_cache] savepath   = {savepath}", flush=True)
    print(f"[prepare_eval_cache] cache_path = {cache_path}", flush=True)

    if os.path.isfile(cache_path) and not args.force:
        print(f"[prepare_eval_cache] cache already exists; pass --force to rebuild.", flush=True)
        return 0

    # Load config
    config_module = __import__(f"config.{args.config}", fromlist=[args.config])
    mode_args = config_module.mode_to_args.get(args.mode, {})
    base_args = config_module.base["diffusion"]
    merged = {**base_args, **mode_args}

    # Cfg namespace
    class _Args: pass
    cfg = _Args()
    for k, v in merged.items():
        setattr(cfg, k, v)
    cfg.device = "cpu"  # we only need the dataset + normalizer; avoid GPU init

    if not getattr(cfg, "multitask", False):
        raise RuntimeError(
            f"Config '{args.config}' (mode='{args.mode}') does not have multitask=True; "
            f"this prep script is only meant for multitask configs."
        )

    # Build the multitask dataset (this is the expensive step we're caching).
    import diffuser.utils as utils
    print("[prepare_eval_cache] building MultitaskGoalDataset (one-time, ~3-5 min)...", flush=True)
    t0 = time.time()
    dataset_config = utils.Config(
        cfg.loader,
        savepath=None,
        dataset_path=getattr(cfg, "override_dataset_path", None),
        dataset_name=cfg.dataset,
        horizon=cfg.horizon,
        obs_only=getattr(cfg, "obs_only", False),
        action_only=getattr(cfg, "action_only", False),
        normalizer=cfg.normalizer,
        particle_normalizer=cfg.particle_normalizer,
        preprocess_fns=cfg.preprocess_fns,
        use_padding=cfg.use_padding,
        max_path_length=cfg.max_path_length,
        overfit=False,
        single_view=(getattr(cfg, "input_type", "dlp") == "dlp" and not cfg.multiview),
        action_z_scale=getattr(cfg, "action_z_scale", 1.0),
        use_gripper_obs=getattr(cfg, "use_gripper_obs", False),
        use_bg_obs=getattr(cfg, "use_bg_obs", False),
        task_entries=getattr(cfg, "task_entries", None),
        max_demos_per_task=getattr(cfg, "max_demos_per_task", None),
    )
    dataset = dataset_config()
    elapsed = time.time() - t0
    print(f"[prepare_eval_cache] dataset built in {elapsed:.1f}s "
          f"(observation_dim={dataset.observation_dim}, action_dim={dataset.action_dim}, "
          f"gripper_dim={getattr(dataset, 'gripper_dim', 0)}, bg_dim={getattr(dataset, 'bg_dim', 0)})",
          flush=True)

    # Strip raw fit data (`.X`) from each sub-normalizer. It's only used in
    # __init__ to compute means/stds; normalize/unnormalize never read it.
    # Dropping it shrinks the cache from ~1.4 GB to ~kB so 12 parallel workers
    # cost essentially nothing in RAM to load.
    _stripped = 0
    if hasattr(dataset.normalizer, "normalizers"):
        for _name, _sub in dataset.normalizer.normalizers.items():
            if hasattr(_sub, "X"):
                try:
                    _sub.X = None
                    _stripped += 1
                except Exception:
                    pass
    print(f"[prepare_eval_cache] stripped .X from {_stripped} sub-normalizer(s) "
          f"to shrink cache size", flush=True)

    payload = {
        "version":         CACHE_VERSION,
        "created":         datetime.now().isoformat(),
        "config":          args.config,
        "mode":            args.mode,
        "savepath":        savepath,
        "normalizer":      dataset.normalizer,
        "observation_dim": int(dataset.observation_dim),
        "action_dim":      int(dataset.action_dim),
        "gripper_dim":     int(getattr(dataset, "gripper_dim", 0)),
        "bg_dim":          int(getattr(dataset, "bg_dim", 0)),
        "particle_dim":    int(getattr(dataset, "particle_dim", 10)),
        "horizon":         int(getattr(dataset, "horizon", cfg.horizon)),
        "action_z_scale":  float(getattr(dataset, "action_z_scale", 1.0)),
        "max_path_length": int(getattr(dataset, "max_path_length", cfg.max_path_length)),
        "n_tasks":         int(getattr(cfg, "n_tasks", 1)),
        "task_names":      list(getattr(cfg, "task_names", []) or []),
    }

    os.makedirs(savepath, exist_ok=True)
    tmp_path = cache_path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, cache_path)

    size_mb = os.path.getsize(cache_path) / (1024 * 1024)
    print(f"[prepare_eval_cache] wrote {cache_path}  ({size_mb:.2f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
