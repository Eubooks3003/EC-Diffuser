"""
2D one-shot helper: build the (multitask) dataset once and freeze the bits eval
needs into <savepath>/eval_cache.pkl.

Mirrors prepare_eval_cache.py from the 3D repo but uses the 2D loader's
dataset_config args (clip_model_name, lang_pooled, max_lang_tokens, etc.).

Usage (mirrors eval_paper_rlbench_2d.py args):
    cd /home/ellina/Desktop/EC-Diffuser-2D
    python diffuser/scripts/prepare_eval_cache_2d.py \\
        --config config.rlbench_multitask_keypose_multiview_dlp \\
        --dataset multitask --num_entity 16 --input_type dlp --seed 42

Idempotent: re-running with --force overwrites <savepath>/eval_cache.pkl.
"""
import argparse
import copy
import os
import pickle
import sys
from datetime import datetime

import diffuser.utils as utils

CACHE_FILENAME = "eval_cache.pkl"
CACHE_VERSION = 1


def _build_args(raw_argv):
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--out", type=str, default=None,
                     help="Output path (default: <savepath>/" + CACHE_FILENAME + ")")
    pre.add_argument("--force", action="store_true",
                     help="Overwrite an existing cache file")
    ours, rest = pre.parse_known_args(raw_argv)

    from diffuser.utils.args import ArgsParser
    sys.argv = [sys.argv[0]] + rest
    args = ArgsParser().parse_args("diffusion")
    args._out = ours.out
    args._force = ours.force
    return args


def _strip_raw_X(normalizer):
    """Drop the giant `self.X` array kept by the Normalizer constructor.

    Eval-time normalize/unnormalize only need (mins, maxs, means, stds, z, x_dim);
    keeping `self.X` means each worker carries the entire training dataset in RAM
    after unpickling.
    """
    snap = copy.copy(normalizer)
    if hasattr(snap, "X"):
        try:
            delattr(snap, "X")
        except Exception:
            snap.X = None
    return snap


def _snapshot_dataset_normalizer(dataset_normalizer):
    """Snapshot DatasetNormalizer in a form safe to pickle without raw data."""
    snap = copy.copy(dataset_normalizer)
    snap.normalizers = {
        key: _strip_raw_X(n) for key, n in dataset_normalizer.normalizers.items()
    }
    return snap


def main(argv):
    args = _build_args(argv)

    out_path = args._out or os.path.join(args.savepath, CACHE_FILENAME)
    if os.path.exists(out_path) and not args._force:
        print(f"[prepare_eval_cache_2d] cache already exists: {out_path}", flush=True)
        print(f"[prepare_eval_cache_2d] re-run with --force to overwrite", flush=True)
        return

    print(f"[prepare_eval_cache_2d] savepath  = {args.savepath}", flush=True)
    print(f"[prepare_eval_cache_2d] dataset   = {args.dataset}", flush=True)
    print(f"[prepare_eval_cache_2d] config    = {args.config}", flush=True)
    print(f"[prepare_eval_cache_2d] out       = {out_path}", flush=True)

    print(f"[prepare_eval_cache_2d] building dataset (this is the slow part)...",
          flush=True)
    t0 = datetime.now()
    dataset_config = utils.Config(
        args.loader,
        savepath=(args.savepath, "dataset_config.pkl"),
        env="",
        dataset_path=args.override_dataset_path,
        horizon=args.horizon,
        normalizer=args.normalizer,
        particle_normalizer=args.particle_normalizer,
        preprocess_fns=args.preprocess_fns,
        use_padding=args.use_padding,
        max_path_length=args.max_path_length,
        dataset_name=args.dataset,
        obs_only=args.obs_only,
        action_only=args.action_only,
        action_z_scale=getattr(args, "action_z_scale", 1.0),
        use_gripper_obs=getattr(args, "use_gripper_obs", False),
        use_bg_obs=getattr(args, "use_bg_obs", False),
        overfit=getattr(args, "overfit", False),
        gripper_state_mask_ratio=getattr(args, "gripper_state_mask_ratio", 0.0),
        single_view=(
            args.input_type == "dlp"
            and not args.multiview
            and getattr(args, "use_views", None) is None
        ),
        clip_model_name=getattr(args, "clip_model_name", "openai/clip-vit-base-patch32"),
        lang_pooled=getattr(args, "lang_pooled", False),
        max_lang_tokens=getattr(args, "max_lang_tokens", 32),
        lang_device=getattr(args, "lang_device", "cpu"),
        use_views=getattr(args, "use_views", None),
        num_source_views=getattr(args, "num_source_views", None),
        action_normalizer=getattr(args, "action_normalizer", None),
        keypose_mode=getattr(args, "keypose_mode", False),
    )
    dataset = dataset_config()
    elapsed = (datetime.now() - t0).total_seconds()
    print(f"[prepare_eval_cache_2d] dataset built in {elapsed:.1f}s", flush=True)

    payload = {
        "version":          CACHE_VERSION,
        "created":          datetime.now().isoformat(),
        "config":           args.config,
        "dataset_name":     args.dataset,
        "dataset_paths":    list(args.override_dataset_path) if isinstance(args.override_dataset_path, (list, tuple)) else [args.override_dataset_path],

        "normalizer":       _snapshot_dataset_normalizer(dataset.normalizer),

        "observation_dim":  int(dataset.observation_dim),
        "action_dim":       int(dataset.action_dim),
        "gripper_dim":      int(getattr(dataset, "gripper_dim", 0)),
        "bg_dim":           int(getattr(dataset, "bg_dim", 0)),

        "horizon":          int(getattr(dataset, "horizon", args.horizon)),
        "action_z_scale":   float(getattr(dataset, "action_z_scale", 1.0)),
        "use_gripper_obs":  bool(getattr(dataset, "use_gripper_obs", False)),
        "use_bg_obs":       bool(getattr(dataset, "use_bg_obs", False)),
        "keypose_mode":     bool(getattr(args, "keypose_mode", False)),
    }

    print(f"[prepare_eval_cache_2d] dims: obs={payload['observation_dim']} "
          f"act={payload['action_dim']} grip={payload['gripper_dim']} "
          f"bg={payload['bg_dim']}", flush=True)
    print(f"[prepare_eval_cache_2d] normalizer keys: "
          f"{sorted(payload['normalizer'].normalizers.keys())}", flush=True)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"[prepare_eval_cache_2d] wrote {out_path}  ({size_mb:.2f} MB)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
