"""
Paper-style multi-seed eval for the RLBench 3D voxel-DLP language-conditioned policy.

Runs N rollouts per seed across a list of seeds, saves videos, and reports the
mean +/- std success rate. Mirrors the 2D eval_paper_rlbench_2d.py in the
EC-Diffuser-2D repo, adapted for the 3D voxel DLP pipeline used here (voxel
encoder + live voxelization inside RLBenchDLPEnv).

Usage:
    xvfb-run -a python diffuser/scripts/eval_paper_rlbench_3d.py \
        --config config.rlbench_open_drawer_dlp \
        --dataset open_drawer \
        --num_entity 16 --input_type dlp --seed 42 \
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

# Make lpwm-dev importable (same strategy as train.py).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_env_lpwm = os.environ.get("LPWM_DEV")
_cands = ([_env_lpwm] if _env_lpwm else []) + [
    os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "..", "lpwm-dev")),
]
for _p in _cands:
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)
        break

import diffuser.utils as utils


# ---------------------------------------------------------------------------
# 3D voxel DLP loading (same constructor as train.py)
# ---------------------------------------------------------------------------
def _build_dlp_from_cfg(cfg, device, DLPClass):
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


def _load_dlp_lpwm(dlp_cfg_path, dlp_ckpt_path, device):
    from utils.util_func import get_config
    from utils.log_utils import load_checkpoint
    from voxel_models import DLP as DLPClass

    dev = torch.device(device)
    cfg = get_config(dlp_cfg_path)
    model = _build_dlp_from_cfg(cfg, dev, DLPClass)
    _ = load_checkpoint(dlp_ckpt_path, model, None, None, map_location=dev)
    model.eval()
    return model, cfg


# ---------------------------------------------------------------------------
# CLI + savepath-aware ckpt resolution
# ---------------------------------------------------------------------------
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
    pre.add_argument("--max_steps", type=int, default=200,
                     help="Max env steps per episode (default: 200)")
    pre.add_argument("--video_episodes", type=int, default=5,
                     help="Keep videos for only the first N episodes per seed; "
                          "set to 0 to keep all (default: 5)")
    pre.add_argument("--video_fps", type=int, default=10,
                     help="Video FPS (default: 10)")
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
    args._video_fps = ours.video_fps
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
    """Delete files named 'epNN_{success,fail}.mp4' with ep >= keep_first_n."""
    if keep_first_n <= 0 or not os.path.isdir(video_dir):
        return
    import re
    pat = re.compile(r"^ep(\d+)(?:_(?:success|fail))?\.mp4$")
    for fname in os.listdir(video_dir):
        m = pat.match(fname)
        if not m:
            continue
        if int(m.group(1)) >= keep_first_n:
            try:
                os.remove(os.path.join(video_dir, fname))
            except OSError:
                pass


def main(argv):
    args = _build_args(argv)
    print(f"[paper_eval_rlbench_3d] savepath = {args.savepath}", flush=True)
    print(f"[paper_eval_rlbench_3d] seeds={args._seeds} "
          f"n_rollouts/seed={args._n_rollouts} "
          f"max_steps={args._max_steps} "
          f"video_episodes={args._video_episodes}", flush=True)

    # ---- dataset ----------------------------------------------------------
    dataset_config = utils.Config(
        args.loader,
        savepath=(args.savepath, "dataset_config.pkl"),
        dataset_path=args.override_dataset_path,
        dataset_name=args.dataset,
        horizon=args.horizon,
        obs_only=args.obs_only,
        action_only=args.action_only,
        normalizer=args.normalizer,
        particle_normalizer=args.particle_normalizer,
        preprocess_fns=args.preprocess_fns,
        use_padding=args.use_padding,
        max_path_length=args.max_path_length,
        overfit=getattr(args, "overfit", False),
        single_view=(args.input_type == "dlp" and not args.multiview),
        action_z_scale=getattr(args, "action_z_scale", 1.0),
        use_gripper_obs=getattr(args, "use_gripper_obs", False),
        use_bg_obs=getattr(args, "use_bg_obs", False),
        keypose_mode=getattr(args, "keypose_mode", False),
    )
    dataset = dataset_config()
    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim
    gripper_dim = getattr(dataset, "gripper_dim", 0)
    bg_dim = getattr(dataset, "bg_dim", 0)
    print(f"[paper_eval_rlbench_3d] obs_dim={observation_dim} action_dim={action_dim} "
          f"gripper_dim={gripper_dim} bg_dim={bg_dim}", flush=True)

    # ---- model + diffusion ------------------------------------------------
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
        use_cond_tokens=getattr(args, 'keypose_mode', False),
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
        obs_only=args.obs_only,
        action_only=args.action_only,
        keypose_mode=getattr(args, 'keypose_mode', False),
    )
    model = model_config()
    diffusion = diffusion_config(model)
    diffusion.to(args.device).eval()

    # ---- load checkpoint --------------------------------------------------
    ckpt_path = _resolve_ckpt(args)
    print(f"[paper_eval_rlbench_3d] loading checkpoint: {ckpt_path}", flush=True)
    ckpt_data = torch.load(ckpt_path, map_location=args.device)

    # ---- load 3D voxel DLP ------------------------------------------------
    dlp_ckpt = getattr(args, "dlp_ckpt", None)
    dlp_cfg_path = getattr(args, "dlp_cfg", None)
    if dlp_ckpt is None or dlp_cfg_path is None:
        raise RuntimeError("3D RLBench eval requires config 'dlp_ckpt' and 'dlp_cfg'")
    print(f"[paper_eval_rlbench_3d] loading DLP cfg={dlp_cfg_path} ckpt={dlp_ckpt}", flush=True)
    dlp_model, dlp_cfg = _load_dlp_lpwm(dlp_cfg_path, dlp_ckpt, args.device)

    # ---- trainer shim -----------------------------------------------------
    # The 3D eval_rlbench_rollouts calls wandb.log(...) internally, so init
    # wandb in disabled mode — the logs go nowhere, but the calls succeed.
    import wandb
    os.environ.setdefault("WANDB_MODE", "disabled")
    wandb.init(project="paper_eval_rlbench_3d", mode="disabled",
               config={"ckpt_path": ckpt_path})

    # Build the renderer shim the trainer expects (optional: catch failures).
    renderer = None
    try:
        render_config = utils.Config(
            args.renderer,
            savepath=(args.savepath, "render_config.pkl"),
            env=None,
            particle_dim=args.features_dim,
        )
        renderer = render_config()
        renderer.latent_rep_model = dlp_model
    except Exception as e:
        print(f"[paper_eval_rlbench_3d] renderer skipped: {type(e).__name__}: {e}", flush=True)

    trainer_config = utils.Config(
        utils.Trainer,
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
    trainer = trainer_config(diffusion, dataset, renderer)

    # Populate both the base and EMA models from the checkpoint.
    state = ckpt_data.get("ema", ckpt_data.get("model", ckpt_data))
    if "model" in ckpt_data:
        trainer.model.load_state_dict(ckpt_data["model"], strict=False)
    if "ema" in ckpt_data:
        trainer.ema_model.load_state_dict(ckpt_data["ema"], strict=False)
    else:
        trainer.ema_model.load_state_dict(state, strict=False)

    try:
        ckpt_step = int(ckpt_data.get("step",
                                      os.path.basename(ckpt_path).rsplit("_step", 1)[-1].split(".")[0]))
    except Exception:
        ckpt_step = 0

    # ---- env factory (cached: CoppeliaSim cannot re-launch cleanly) ------
    _env_cache = {"env": None}

    def make_env_fn():
        if _env_cache["env"] is not None:
            return _env_cache["env"]
        from diffuser.envs.rlbench_dlp_wrapper import RLBenchDLPEnv
        env = RLBenchDLPEnv(
            task_name=args.dataset,
            dlp_model=dlp_model,
            dlp_cfg=dlp_cfg,
            cams=getattr(args, "rlbench_eval_cams",
                         ["front", "overhead", "left_shoulder", "right_shoulder"]),
            image_size=int(getattr(args, "rlbench_eval_image_size", 128)),
            headless=True,
            episode_length=int(args._max_steps),
            device=args.device,
        )
        _env_cache["env"] = env
        return env

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

    # ---- output path (created up-front so partial results are always saved)
    out_dir = args._output_dir or os.path.join(args.savepath, "paper_eval")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_name = os.path.basename(ckpt_path).replace(".pt", "")
    out_path = os.path.join(
        out_dir,
        f"eval_{ckpt_name}_seeds{'-'.join(map(str, args._seeds))}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )

    def _write_json(per_seed_list, status):
        rates = np.array([r["success_rate"] for r in per_seed_list]) if per_seed_list else np.array([])
        tot_n = sum(r["n_episodes"] for r in per_seed_list)
        tot_s = sum(r["n_successes"] for r in per_seed_list)
        payload = {
            "status": status,
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
            "per_seed": per_seed_list,
            "mean_success_rate": float(rates.mean()) if len(rates) else 0.0,
            "std_success_rate": float(rates.std(ddof=0)) if len(rates) else 0.0,
            "overall_success_rate": (tot_s / tot_n) if tot_n else 0.0,
            "total_rollouts": tot_n,
            "total_successes": tot_s,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2,
                      default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))
        return payload

    # ---- seed loop --------------------------------------------------------
    per_seed = []

    for seed in args._seeds:
        print("\n" + "=" * 72, flush=True)
        print(f"[paper_eval_rlbench_3d] SEED {seed}  "
              f"({args._n_rollouts} episodes, max_steps={args._max_steps})", flush=True)
        print("=" * 72, flush=True)

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Route this seed's videos to eval_results/seed_{S}/
        video_dir = os.path.join(args.savepath, "eval_results", f"seed_{seed}")

        sim_stats = trainer.eval_rlbench_rollouts(
            make_env_fn=make_env_fn,
            make_policy_fn=make_policy_fn,
            n_episodes=args._n_rollouts,
            max_steps=args._max_steps,
            exe_steps=int(getattr(args, "exe_steps", 1)),
            video_fps=args._video_fps,
            wandb_step=seed,
            log_voxel_viz=False,
            video_dir_override=video_dir,
        )

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
        print(f"[paper_eval_rlbench_3d] seed {seed}: "
              f"{n_succ}/{n_ep} = {sr*100:.1f}%", flush=True)

        # Persist partial results after every seed.
        _write_json(per_seed, status="partial")
        print(f"[paper_eval_rlbench_3d] partial results -> {out_path}", flush=True)

    # ---- aggregate --------------------------------------------------------
    rates = np.array([r["success_rate"] for r in per_seed])
    total_n = sum(r["n_episodes"] for r in per_seed)
    total_succ = sum(r["n_successes"] for r in per_seed)
    mean_sr = float(rates.mean()) if len(rates) else 0.0
    std_sr = float(rates.std(ddof=0)) if len(rates) else 0.0
    overall_sr = total_succ / total_n if total_n else 0.0

    print("\n" + "=" * 72, flush=True)
    print("[paper_eval_rlbench_3d] FINAL RESULTS", flush=True)
    print("=" * 72, flush=True)
    for r in per_seed:
        print(f"  seed {r['seed']:>6}: "
              f"{r['n_successes']:>3}/{r['n_episodes']:<3} "
              f"= {r['success_rate']*100:5.1f}%", flush=True)
    print(f"  mean +/- std : {mean_sr*100:5.1f}% +/- {std_sr*100:.1f}%", flush=True)
    print(f"  overall      : {total_succ}/{total_n} = {overall_sr*100:5.1f}%", flush=True)
    print("=" * 72, flush=True)

    # ---- save JSON (final) -----------------------------------------------
    _write_json(per_seed, status="complete")
    print(f"[paper_eval_rlbench_3d] saved results: {out_path}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
