"""
Paper-style multi-seed eval for the RLBench 2D-DLP language-conditioned policy.

Runs N rollouts per seed across a list of seeds, saves videos, and reports the
mean +/- std success rate (mirrors scripts/eval_paper.py for MimicGen).

Usage:
    xvfb-run -a python diffuser/scripts/eval_paper_rlbench_2d.py \
        --config config.rlbench_open_drawer_dlp \
        --dataset open_drawer \
        --num_entity 16 --input_type dlp \
        --ckpt_step 1820000 \
        --n_rollouts 50 \
        --seeds 42,123,456 \
        --video_episodes 5
"""
import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch

import diffuser.utils as utils


def _build_args(raw_argv):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--ckpt_step", default="latest",
                     help="Checkpoint step to load: 'latest' or an int (e.g. 1820000)")
    pre.add_argument("--ckpt_path", default=None,
                     help="Explicit ckpt .pt path (overrides --ckpt_step lookup)")
    pre.add_argument("--n_rollouts", type=int, default=50,
                     help="Number of rollouts per seed (default: 50)")
    pre.add_argument("--seeds", type=str, default="42,123,456",
                     help="Comma-separated list of seeds (default: 42,123,456)")
    pre.add_argument("--max_steps", type=int, default=400,
                     help="Max env steps per episode (default: 400)")
    pre.add_argument("--video_episodes", type=int, default=5,
                     help="Keep videos for only the first N episodes per seed; "
                          "set to 0 to keep all (default: 5)")
    pre.add_argument("--output_dir", type=str, default=None,
                     help="Where to write eval_results.json (default: "
                          "<savepath>/paper_eval/)")
    ours, rest = pre.parse_known_args(raw_argv)

    from diffuser.utils.args import ArgsParser
    sys.argv = [sys.argv[0]] + rest
    args = ArgsParser().parse_args("diffusion")
    args._ckpt_step = ours.ckpt_step
    args._ckpt_path = ours.ckpt_path
    args._n_rollouts = ours.n_rollouts
    args._seeds = [int(s) for s in ours.seeds.split(",") if s.strip()]
    args._max_steps = ours.max_steps
    args._video_episodes = ours.video_episodes
    args._output_dir = ours.output_dir
    return args


def _resolve_ckpt(args):
    if args._ckpt_path:
        return args._ckpt_path
    ckpt_dir = os.path.join(args.savepath, "ckpt")
    files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    if not files:
        raise RuntimeError(f"No .pt checkpoint under {ckpt_dir}")
    if args._ckpt_step == "latest":
        def _step_of(f):
            try:
                return int(f.rsplit("_step", 1)[-1].split(".")[0])
            except Exception:
                return -1
        files.sort(key=_step_of)
        return os.path.join(ckpt_dir, files[-1])
    matches = [f for f in files if f.endswith(f"_step{args._ckpt_step}.pt")]
    if not matches:
        raise RuntimeError(f"No ckpt for step {args._ckpt_step} in {ckpt_dir}")
    return os.path.join(ckpt_dir, matches[0])


def _trim_videos(video_dir, keep_first_n):
    """Delete per-episode video files with ep index >= keep_first_n."""
    if keep_first_n <= 0 or not os.path.isdir(video_dir):
        return
    import re
    ep_pat = re.compile(r"^ep(\d+)_")
    for fname in os.listdir(video_dir):
        m = ep_pat.match(fname)
        if not m:
            continue
        if int(m.group(1)) >= keep_first_n:
            try:
                os.remove(os.path.join(video_dir, fname))
            except OSError:
                pass


def main(argv):
    args = _build_args(argv)
    print(f"[paper_eval_rlbench] savepath = {args.savepath}", flush=True)
    print(f"[paper_eval_rlbench] seeds={args._seeds} "
          f"n_rollouts/seed={args._n_rollouts} "
          f"max_steps={args._max_steps} "
          f"video_episodes={args._video_episodes}", flush=True)

    # -------- dataset (mirrors eval_rlbench.py) ----------------------------
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
        max_demos=getattr(args, "max_demos", None),
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
    )
    dataset = dataset_config()
    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim
    gripper_dim = getattr(dataset, "gripper_dim", 0)
    bg_dim = getattr(dataset, "bg_dim", 0)

    # -------- model + diffusion --------------------------------------------
    model_config = utils.Config(
        args.model,
        savepath=(args.savepath, "model_config.pkl"),
        features_dim=args.features_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        projection_dim=args.projection_dim,
        n_head=args.n_heads,
        n_layer=args.n_layers,
        dropout=args.dropout,
        block_size=args.horizon,
        positional_bias=args.positional_bias,
        max_particles=args.max_particles,
        multiview=args.multiview,
        device=args.device,
        gripper_dim=gripper_dim,
        bg_dim=bg_dim,
        lang_dim=getattr(args, "lang_dim", 0),
        act_pos_dim=getattr(args, "act_pos_dim", 3),
        act_rot_dim=getattr(args, "act_rot_dim", 3),
        act_grip_dim=getattr(args, "act_grip_dim", 1),
        prop_pos_dim=getattr(args, "prop_pos_dim", 3),
        prop_rot_dim=getattr(args, "prop_rot_dim", 6),
        prop_grip_dim=getattr(args, "prop_grip_dim", 1),
    )
    diffusion_config = utils.Config(
        args.diffusion,
        savepath=(args.savepath, "diffusion_config.pkl"),
        horizon=args.horizon,
        observation_dim=observation_dim,
        action_dim=action_dim,
        gripper_dim=gripper_dim,
        bg_dim=bg_dim,
        n_timesteps=args.n_diffusion_steps,
        loss_type=args.loss_type,
        clip_denoised=args.clip_denoised,
        predict_epsilon=args.predict_epsilon,
        action_weight=args.action_weight,
        loss_weights=args.loss_weights,
        loss_discount=args.loss_discount,
        device=args.device,
    )
    model = model_config()
    diffusion = diffusion_config(model)
    diffusion.to(args.device).eval()

    # -------- load checkpoint ---------------------------------------------
    ckpt_path = _resolve_ckpt(args)
    print(f"[paper_eval_rlbench] loading checkpoint: {ckpt_path}", flush=True)
    sd = torch.load(ckpt_path, map_location=args.device)
    state = sd.get("ema", sd.get("model", sd))
    diffusion.load_state_dict(state, strict=False)

    # -------- DLP encoder (for live encoding) ------------------------------
    dlp_ckpt = args.dlp_ckpt
    dlp_cfg_path = args.dlp_cfg
    _dlp_ctor_eval = getattr(args, "dlp_ctor", "models:DLP")
    print(f"[paper_eval_rlbench] loading DLP from {dlp_ckpt}", flush=True)

    def _build_dlp_2d_from_cfg(cfg, dev, DLPClass):
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
        ).to(dev)
        m.eval()
        return m

    from utils.util_func import get_config
    dev = torch.device(args.device)
    dlp_cfg = get_config(dlp_cfg_path)
    if "voxel" in _dlp_ctor_eval.lower():
        raise RuntimeError(f"3D DLP not supported here; dlp_ctor={_dlp_ctor_eval}")
    from models import DLP as _DLPClass
    dlp_model = _build_dlp_2d_from_cfg(dlp_cfg, dev, _DLPClass)
    dlp_model.load_state_dict(torch.load(dlp_ckpt, map_location=dev, weights_only=False))
    dlp_model.eval()

    _cams = getattr(args, "rlbench_cams",
                    ["front", "overhead", "left_shoulder", "right_shoulder"])
    _img_size = int(getattr(args, "rlbench_image_size", 128))

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
    def _dlp_encode_fn(rgbs_np):
        rgbs = torch.from_numpy(np.asarray(rgbs_np)).float() / 255.0
        rgbs = rgbs.permute(0, 3, 1, 2).to(args.device)
        if rgbs.shape[-1] != _img_size:
            rgbs = torch.nn.functional.interpolate(
                rgbs, size=(_img_size, _img_size), mode="bilinear", align_corners=False
            )
        per_view_toks = []; per_view_bg = []
        for vi in range(rgbs.shape[0]):
            chunk = rgbs[vi:vi + 1]
            enc = dlp_model.encode_all(chunk, deterministic=True)
            toks_v, bg_v = _pack_tokens_2d(enc)
            per_view_toks.append(toks_v); per_view_bg.append(bg_v)
        toks = torch.cat(per_view_toks, dim=0)
        bg = torch.cat(per_view_bg, dim=0)
        V, K, Dtok = toks.shape
        return {
            "tokens": toks.reshape(V * K, Dtok).cpu().numpy(),
            "bg": bg.reshape(-1).cpu().numpy(),
        }

    # -------- trainer shim --------------------------------------------------
    from diffuser.utils.training import Trainer
    trainer_config = utils.Config(
        Trainer,
        savepath=(args.savepath, "trainer_paper_eval.pkl"),
        train_batch_size=args.batch_size,
        train_lr=args.learning_rate,
        gradient_accumulate_every=args.gradient_accumulate_every,
        ema_decay=args.ema_decay,
        sample_freq=10**9,
        save_freq=10**9,
        label_freq=10**9,
        save_parallel=False,
        results_folder=args.savepath,
        bucket=args.bucket,
        n_reference=0,
    )
    trainer = trainer_config(diffusion, dataset, None)
    trainer.ema_model.load_state_dict(state, strict=False)

    # Record ckpt step for bookkeeping (doesn't affect video dir: we override below)
    try:
        ckpt_step = int(os.path.basename(ckpt_path).rsplit("_step", 1)[-1].split(".")[0])
    except Exception:
        ckpt_step = 0

    # -------- env factory (cached: RLBench cannot re-launch CoppeliaSim) ----
    _env_cache = {"env": None}

    def make_env_fn():
        if _env_cache["env"] is not None:
            return _env_cache["env"]
        from diffuser.envs.rlbench_dlp_wrapper import RLBenchDLPEnv
        _env = RLBenchDLPEnv(
            task_name=args.dataset,
            dlp_encode_fn=_dlp_encode_fn,
            cams=_cams,
            image_size=_img_size,
            headless=bool(getattr(args, "rlbench_headless", True)),
            episode_length=int(args._max_steps),
            delta_actions=not bool(getattr(args, "use_absolute_actions", True)),
        )
        _env._dlp_model = dlp_model
        _env_cache["env"] = _env
        return _env

    def make_policy_fn():
        from diffuser.sampling import LanguageConditionedPolicy
        return LanguageConditionedPolicy(
            diffusion_model=trainer.ema_model,
            normalizer=dataset.normalizer,
            preprocess_fns=[],
            clip_model_name=getattr(args, "clip_model_name", "openai/clip-vit-base-patch32"),
            lang_pooled=bool(getattr(args, "lang_pooled", False)),
            max_lang_tokens=int(getattr(args, "max_lang_tokens", 32)),
        )

    # Auto-populate diagnostic flags (same as eval_rlbench.py).
    if getattr(args, "save_gt_video", False):
        os.environ.setdefault("ECDIFF_SAVE_GT_VIDEO", "1")
    _demo_root_cfg = getattr(args, "demo_dataset_root", None)
    if _demo_root_cfg:
        os.environ.setdefault("ECDIFF_DEMO_ROOT", str(_demo_root_cfg))
    if getattr(args, "save_imagined", False):
        os.environ.setdefault("ECDIFF_SAVE_IMAGINED", "1")
    if getattr(args, "save_imagined_recon", False):
        os.environ.setdefault("ECDIFF_SAVE_IMAGINED_RECON", "1")

    # -------- seed loop -----------------------------------------------------
    per_seed = []
    for seed in args._seeds:
        print("\n" + "=" * 72, flush=True)
        print(f"[paper_eval_rlbench] SEED {seed}  "
              f"({args._n_rollouts} episodes, max_steps={args._max_steps})", flush=True)
        print("=" * 72, flush=True)

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Route this seed's videos to eval_videos/step_seed{S}_ckpt{step}/
        trainer.step = seed  # used by trainer as the video-dir suffix
        video_dir = os.path.join(args.savepath, "eval_videos", f"step_{seed}")

        sim_stats = trainer.eval_rlbench_rollouts(
            make_env_fn=make_env_fn,
            make_policy_fn=make_policy_fn,
            n_episodes=args._n_rollouts,
            max_steps=args._max_steps,
            task_name=args.dataset,
            exe_steps=getattr(args, "exe_steps", 1),
        )

        # Optionally prune extra episode videos
        _trim_videos(video_dir, args._video_episodes)

        sr = float(sim_stats.get("sim/success_rate", 0.0))
        n_ep = int(sim_stats.get("sim/n_episodes", 0))
        n_succ = int(round(sr * n_ep))
        per_seed.append({
            "seed": seed,
            "n_episodes": n_ep,
            "n_successes": n_succ,
            "success_rate": sr,
            "avg_length": float(sim_stats.get("sim/avg_len", 0.0)),
            "video_dir": video_dir,
            "sim_stats": sim_stats,
        })
        print(f"[paper_eval_rlbench] seed {seed}: "
              f"{n_succ}/{n_ep} = {sr*100:.1f}%", flush=True)

    # -------- aggregate -----------------------------------------------------
    rates = np.array([r["success_rate"] for r in per_seed])
    total_n = sum(r["n_episodes"] for r in per_seed)
    total_succ = sum(r["n_successes"] for r in per_seed)
    mean_sr = float(rates.mean()) if len(rates) else 0.0
    std_sr = float(rates.std(ddof=0)) if len(rates) else 0.0
    overall_sr = total_succ / total_n if total_n else 0.0

    print("\n" + "=" * 72, flush=True)
    print("[paper_eval_rlbench] FINAL RESULTS", flush=True)
    print("=" * 72, flush=True)
    for r in per_seed:
        print(f"  seed {r['seed']:>6}: "
              f"{r['n_successes']:>3}/{r['n_episodes']:<3} "
              f"= {r['success_rate']*100:5.1f}%", flush=True)
    print(f"  mean +/- std : {mean_sr*100:5.1f}% +/- {std_sr*100:.1f}%", flush=True)
    print(f"  overall      : {total_succ}/{total_n} = {overall_sr*100:5.1f}%", flush=True)
    print("=" * 72, flush=True)

    # -------- save JSON -----------------------------------------------------
    out_dir = args._output_dir or os.path.join(args.savepath, "paper_eval")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_name = os.path.basename(ckpt_path).replace(".pt", "")
    out_path = os.path.join(
        out_dir,
        f"eval_{ckpt_name}_seeds{'-'.join(map(str, args._seeds))}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    results = {
        "config": getattr(args, "config", None),
        "dataset": args.dataset,
        "ckpt_path": ckpt_path,
        "ckpt_step": ckpt_step,
        "n_rollouts_per_seed": args._n_rollouts,
        "seeds": args._seeds,
        "max_steps": args._max_steps,
        "exe_steps": int(getattr(args, "exe_steps", 1)),
        "video_episodes": args._video_episodes,
        "savepath": args.savepath,
        "timestamp": datetime.now().isoformat(),
        "per_seed": per_seed,
        "mean_success_rate": mean_sr,
        "std_success_rate": std_sr,
        "overall_success_rate": overall_sr,
        "total_rollouts": total_n,
        "total_successes": total_succ,
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))
    print(f"[paper_eval_rlbench] saved results: {out_path}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
