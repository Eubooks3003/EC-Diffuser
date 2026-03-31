#!/usr/bin/env python
"""
Evaluation script for paper results — 2D Isaac PandaPush setup.

Runs N rollouts with multiple seeds and saves success rates.
This is the 2D counterpart of eval_paper.py (which targets MimicGen/3D tasks).

Usage:
    python eval_paper_2d.py \
        --config pandapush_pint \
        --mode 2C_dlp \
        --loadpath data/pandapush/diffusion/pandapush_pint/2C_dlp_... \
        --epoch latest \
        --n_rollouts 50 \
        --seeds 42,123,456 \
        --output_dir ./eval_results_2d

    # Or load a specific .pt checkpoint directly:
    python eval_paper_2d.py \
        --config pandapush_pint \
        --mode 2C_dlp \
        --ckpt_path /path/to/ckpt/state_0_step100000.pt \
        --n_rollouts 50 \
        --seeds 42,123,456

Arguments:
    --config: Config file name (e.g., pandapush_pint)
    --mode: Mode key in config (e.g., 1C_dlp, 2C_dlp, 3C_dlp, 1C_dlp_pusht, etc.)
    --loadpath: Path to training output folder (contains ckpt/ and *_config.pkl).
                Mutually exclusive with --ckpt_path.
    --epoch: Which checkpoint epoch to load when using --loadpath (default: latest)
    --ckpt_path: Direct path to a .pt checkpoint file.
                 Mutually exclusive with --loadpath.
    --n_rollouts: Number of rollouts per seed (default: 50)
    --seeds: Comma-separated list of seeds (default: 42,123,456)
    --output_dir: Output directory for results (default: alongside ckpt)
    --device: Device to use (default: cuda:0)
    --exe_steps: Override action-chunking steps (default: from config plan.exe_steps)
    --save_videos: Enable video saving
    --video_episodes: Episodes to record per seed (default: 5)
    --video_fps: Video FPS (default: 15)
    --env_config_dir: Override env config directory (default: from config mode)
    --num_envs: Number of parallel Isaac envs (default: 1, >1 not yet supported for rollouts)
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import argparse
import json
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm

# ---------------------------------------------------------------------------
#  path setup — make EC-Diffuser root, diffuser/, and siblings importable
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DIFFUSER_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
EC_DIFFUSER_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

for p in [EC_DIFFUSER_ROOT, DIFFUSER_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

# lpwm-copy is a sibling of EC-Diffuser and holds the 2D DLP code
for _sibling in ("lpwm-copy", "lpwm-dev"):
    _p = os.path.abspath(os.path.join(EC_DIFFUSER_ROOT, "..", _sibling))
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)


# ---------------------------------------------------------------------------
def _to_uint8(img):
    """Convert image to uint8 format for video saving."""
    img = np.asarray(img)
    if img.ndim == 4 and img.shape[0] == 1:
        img = img[0]
    # CHW -> HWC if needed
    if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        if img.max() <= 1.5:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    return img


# ---------------------------------------------------------------------------
#  env setup
# ---------------------------------------------------------------------------
def setup_2d_env(env_config_dir, device="cuda:0"):
    """
    Create the IsaacPandaPush env wrapped with IsaacPandaPushGoalSB3Wrapper
    using the YAML configs in *env_config_dir*.
    """
    import yaml
    from pathlib import Path
    from dlp_utils import check_config, load_pretrained_rep_model

    config = yaml.safe_load(Path(f"{env_config_dir}/Config.yaml").read_text())
    isaac_env_cfg = yaml.safe_load(
        Path(f"{env_config_dir}/IsaacPandaPushConfig.yaml").read_text()
    )

    # Parse device index
    if ":" in device:
        cuda_idx = int(device.split(":")[-1])
    else:
        cuda_idx = 0
    config["cudaDevice"] = cuda_idx

    check_config(config, isaac_env_cfg)

    from isaac_panda_push_env import IsaacPandaPush

    envs = IsaacPandaPush(
        cfg=isaac_env_cfg,
        rl_device=f"cuda:{cuda_idx}",
        sim_device=f"cuda:{cuda_idx}",
        graphics_device_id=cuda_idx,
        headless=True,
        virtual_screen_capture=False,
        force_render=False,
    )

    latent_rep_model = load_pretrained_rep_model(
        dir_path=config["Model"]["latentRepPath"],
        model_type=config["Model"]["obsMode"],
    )

    from isaac_env_wrappers import IsaacPandaPushGoalSB3Wrapper

    env = IsaacPandaPushGoalSB3Wrapper(
        env=envs,
        obs_mode=config["Model"]["obsMode"],
        n_views=config["Model"]["numViews"],
        latent_rep_model=latent_rep_model,
        latent_classifier=None,
        reward_cfg=config["Reward"]["GT"],
        smorl=False,
        collect_images=True,
    )
    return env, config


# ---------------------------------------------------------------------------
#  rollout loop
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_eval_rollouts(
    policy,
    env,
    n_episodes=50,
    exe_steps=1,
    seed=42,
    save_videos=False,
    video_dir=None,
    video_episodes=5,
    video_fps=15,
):
    """
    Run evaluation rollouts on the 2D Isaac PandaPush environment.

    The policy is a GoalConditionedPolicy that takes {0: achieved_goal, H-1: desired_goal}
    and returns (action, trajectories).

    Returns dict with per-seed results.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    horizon = policy.sample_kwargs.get("horizon", None)
    if horizon is None:
        horizon = policy.diffusion_model.horizon

    successes = []
    success_fracs = []
    returns = []
    lengths = []
    avg_dists = []
    max_dists = []

    # Video saving setup
    if save_videos and video_dir is not None:
        os.makedirs(video_dir, exist_ok=True)
        try:
            import imageio
        except ImportError:
            print("[WARNING] imageio not installed, disabling video saving")
            save_videos = False

    print(f"[eval] seed={seed}, {n_episodes} eps, exe_steps={exe_steps}")

    pbar = tqdm(range(n_episodes), desc=f"Seed {seed}", unit="ep")
    for ep in pbar:
        obs = env.reset()
        ep_ret = 0.0
        done = False

        # Video frame collection
        record_this_episode = save_videos and ep < video_episodes
        frames = []

        t = 0
        while t < env.horizon:
            achieved_goal = obs["achieved_goal"]   # (B, K, Dtok) or (B, n_views, K, Dtok)
            desired_goal = obs["desired_goal"]      # same shape

            conditions = {0: achieved_goal, horizon - 1: desired_goal}
            action_0, samples = policy(conditions, batch_size=1, verbose=False)

            for step in range(exe_steps):
                actions = samples.actions[:, step]  # (B, action_dim)
                obs, reward, done_arr, info = env.step(actions)

                ep_ret += float(np.mean(reward))
                t += 1

                # Capture frame
                if record_this_episode:
                    # info is tuple of dicts for vec envs
                    if isinstance(info, tuple) and len(info) > 0 and "image" in info[0]:
                        frame = info[0]["image"][0]  # first view of first env
                        frames.append(_to_uint8(frame))

                if t >= env.horizon:
                    break

        # Collect final stats from last info
        if isinstance(info, tuple):
            info_0 = info[0]
        else:
            info_0 = info

        goal_frac = float(info_0.get("goal_success_frac", 0.0))
        avg_dist = float(info_0.get("avg_obj_dist", 1.0))
        max_dist = float(info_0.get("max_obj_dist", 1.0))
        success = goal_frac >= 1.0

        successes.append(success)
        success_fracs.append(goal_frac)
        returns.append(ep_ret)
        lengths.append(t)
        avg_dists.append(avg_dist)
        max_dists.append(max_dist)

        pbar.set_postfix(
            sr=f"{np.mean(successes)*100:.0f}%",
            frac=f"{np.mean(success_fracs):.2f}",
            succ=sum(successes),
        )

        # Save video
        if record_this_episode and frames and video_dir is not None:
            import imageio
            status = "success" if success else "fail"
            video_path = os.path.join(video_dir, f"seed{seed}_ep{ep:02d}_{status}.mp4")
            try:
                imageio.mimsave(video_path, frames, fps=video_fps)
            except Exception as e:
                print(f"[WARNING] Failed to save video: {e}")

    # Close environment
    try:
        env.close()
    except Exception:
        pass

    success_rate = float(np.mean(successes))
    mean_frac = float(np.mean(success_fracs))
    print(
        f"[eval] Seed {seed}: success_rate={success_rate:.4f} "
        f"({sum(successes)}/{n_episodes}), mean_frac={mean_frac:.4f}"
    )

    return {
        "seed": seed,
        "n_episodes": n_episodes,
        "successes": [bool(s) for s in successes],
        "success_rate": success_rate,
        "mean_success_frac": mean_frac,
        "avg_return": float(np.mean(returns)),
        "avg_length": float(np.mean(lengths)),
        "avg_obj_dist": float(np.mean(avg_dists)),
        "max_obj_dist": float(np.mean(max_dists)),
    }


# ---------------------------------------------------------------------------
#  main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="EC-Diffuser 2D paper evaluation")

    # Config
    parser.add_argument("--config", type=str, default="pandapush_pint",
                        help="Config file name (default: pandapush_pint)")
    parser.add_argument("--mode", type=str, required=True,
                        help="Mode key in config (e.g., 1C_dlp, 2C_dlp, 3C_dlp)")

    # Checkpoint: two mutually-exclusive ways to specify
    ckpt_group = parser.add_mutually_exclusive_group(required=True)
    ckpt_group.add_argument("--loadpath", type=str, default=None,
                            help="Path to training output folder (with ckpt/ and *_config.pkl)")
    ckpt_group.add_argument("--ckpt_path", type=str, default=None,
                            help="Direct path to a .pt checkpoint file")
    parser.add_argument("--epoch", type=str, default="latest",
                        help="Checkpoint epoch when using --loadpath (default: latest)")

    # Eval settings
    parser.add_argument("--n_rollouts", type=int, default=50,
                        help="Number of rollouts per seed (default: 50)")
    parser.add_argument("--seeds", type=str, default="42,123,456",
                        help="Comma-separated list of seeds (default: 42,123,456)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results (default: alongside ckpt)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--exe_steps", type=int, default=None,
                        help="Override action-chunking exe_steps (default: from config)")
    parser.add_argument("--env_config_dir", type=str, default=None,
                        help="Override env_config_dir (default: from config mode)")

    # Video
    parser.add_argument("--save_videos", action="store_true")
    parser.add_argument("--video_episodes", type=int, default=5)
    parser.add_argument("--video_fps", type=int, default=15)

    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    print(f"\n{'='*60}")
    print(f"EC-Diffuser 2D Paper Evaluation")
    print(f"{'='*60}")
    print(f"Config: {args.config}")
    print(f"Mode: {args.mode}")
    if args.loadpath:
        print(f"Loadpath: {args.loadpath}")
        print(f"Epoch: {args.epoch}")
    else:
        print(f"Checkpoint: {args.ckpt_path}")
    print(f"N rollouts per seed: {args.n_rollouts}")
    print(f"Seeds: {seeds}")
    print(f"Device: {args.device}")
    print(f"Save videos: {args.save_videos}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    #  Load config to get env_config_dir and exe_steps
    # ------------------------------------------------------------------
    config_module = __import__(f"config.{args.config}", fromlist=[args.config])
    mode_args = config_module.mode_to_args.get(args.mode, {})
    plan_args = config_module.base.get("plan", {})

    env_config_dir = args.env_config_dir or mode_args.get(
        "env_config_dir", config_module.base["diffusion"].get("env_config_dir", "env_config/n_cubes")
    )
    exe_steps = args.exe_steps or plan_args.get("exe_steps", 1)

    print(f"env_config_dir: {env_config_dir}")
    print(f"exe_steps: {exe_steps}")

    # ------------------------------------------------------------------
    #  Load diffusion experiment (policy, dataset, etc.)
    # ------------------------------------------------------------------
    import diffuser.utils as utils
    from diffuser.utils.arrays import set_global_device
    set_global_device(args.device)

    if args.loadpath:
        # Use the standard load_diffusion path (loads from *_config.pkl files)
        diffusion_experiment = utils.load_diffusion(
            args.loadpath, "",  # dataset arg unused when configs already contain path
            "",                 # diffusion_loadpath unused
            epoch=args.epoch,
            seed=None,
            is_diffusion=True,
        )
        dataset = diffusion_experiment.dataset
        ema_model = diffusion_experiment.ema
        trainer = diffusion_experiment.trainer
        ckpt_label = f"epoch_{diffusion_experiment.epoch}"
    else:
        # Load from raw checkpoint (same approach as eval_paper.py)
        base_args = config_module.base["diffusion"]
        merged_args = {**base_args, **mode_args}

        class CfgNS:
            pass
        cfg = CfgNS()
        for k, v in merged_args.items():
            setattr(cfg, k, v)
        cfg.device = args.device

        # Dataset
        dataset_path = getattr(cfg, "override_dataset_path", None) or getattr(cfg, "dataset_path", None)
        if dataset_path is None:
            raise RuntimeError(
                "Config must have 'override_dataset_path' or 'dataset_path' "
                "when using --ckpt_path. Use --loadpath instead if you trained "
                "with the standard pipeline."
            )

        dataset_config = utils.Config(
            cfg.loader,
            savepath=None,
            dataset_path=dataset_path,
            dataset_name=getattr(cfg, "dataset", "pandapush"),
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
        )
        dataset = dataset_config()

        observation_dim = dataset.observation_dim
        action_dim = dataset.action_dim
        gripper_dim = getattr(dataset, "gripper_dim", 0)
        bg_dim = getattr(dataset, "bg_dim", 0)

        model_config = utils.Config(
            cfg.model, savepath=None,
            features_dim=cfg.features_dim,
            action_dim=action_dim,
            hidden_dim=cfg.hidden_dim,
            projection_dim=cfg.projection_dim,
            n_head=cfg.n_heads,
            n_layer=cfg.n_layers,
            dropout=cfg.dropout,
            block_size=cfg.horizon,
            positional_bias=cfg.positional_bias,
            max_particles=cfg.max_particles,
            multiview=cfg.multiview,
            device=cfg.device,
            gripper_dim=gripper_dim,
            bg_dim=bg_dim,
        )

        diffusion_config = utils.Config(
            cfg.diffusion, savepath=None,
            horizon=cfg.horizon,
            observation_dim=observation_dim,
            action_dim=action_dim,
            gripper_dim=gripper_dim,
            bg_dim=bg_dim,
            n_timesteps=cfg.n_diffusion_steps,
            loss_type=cfg.loss_type,
            clip_denoised=cfg.clip_denoised,
            predict_epsilon=cfg.predict_epsilon,
            action_weight=cfg.action_weight,
            loss_weights=cfg.loss_weights,
            loss_discount=cfg.loss_discount,
            device=cfg.device,
            obs_only=getattr(cfg, "obs_only", False),
            action_only=getattr(cfg, "action_only", False),
        )

        renderer = None
        try:
            render_config = utils.Config(cfg.renderer, savepath=None, env=None,
                                         particle_dim=cfg.features_dim)
            renderer = render_config()
        except Exception:
            pass

        model = model_config()
        diffusion = diffusion_config(model)
        trainer_config = utils.Config(
            utils.Trainer, savepath=None,
            train_batch_size=cfg.batch_size,
            train_lr=cfg.learning_rate,
            gradient_accumulate_every=cfg.gradient_accumulate_every,
            ema_decay=cfg.ema_decay,
            sample_freq=cfg.sample_freq,
            save_freq=cfg.save_freq,
            label_freq=int(cfg.n_train_steps // cfg.n_saves),
            save_parallel=cfg.save_parallel,
            results_folder=os.path.dirname(args.ckpt_path).replace("/ckpt", ""),
            bucket=cfg.bucket,
            n_reference=cfg.n_reference,
        )
        trainer = trainer_config(diffusion, dataset, renderer)

        # Load checkpoint
        ckpt_data = torch.load(args.ckpt_path, map_location=args.device)
        trainer.step = ckpt_data["step"]
        trainer.model.load_state_dict(ckpt_data["model"])
        trainer.ema_model.load_state_dict(ckpt_data["ema"])
        ema_model = trainer.ema_model
        ckpt_label = os.path.basename(args.ckpt_path).replace(".pt", "")
        print(f"Loaded checkpoint at step {trainer.step}")

    # ------------------------------------------------------------------
    #  Build goal-conditioned policy
    # ------------------------------------------------------------------
    horizon = dataset.horizon if hasattr(dataset, "horizon") else ema_model.horizon
    policy_config = utils.Config(
        "sampling.GoalConditionedPolicy",
        diffusion_model=ema_model,
        normalizer=dataset.normalizer,
        preprocess_fns=[],
        verbose=False,
        horizon=horizon,
    )
    policy = policy_config()

    # ------------------------------------------------------------------
    #  Set up environment (fresh per seed to avoid state leakage)
    # ------------------------------------------------------------------
    # Determine output dir
    if args.output_dir:
        output_dir = args.output_dir
    elif args.ckpt_path:
        output_dir = os.path.join(os.path.dirname(args.ckpt_path), "eval_results_2d")
    else:
        output_dir = os.path.join(args.loadpath, "eval_results_2d")
    os.makedirs(output_dir, exist_ok=True)

    # Video base dir
    video_base_dir = None
    if args.save_videos:
        video_base_dir = os.path.join(output_dir, "videos", ckpt_label)
        os.makedirs(video_base_dir, exist_ok=True)

    # ------------------------------------------------------------------
    #  Run evaluation for each seed
    # ------------------------------------------------------------------
    all_results = []
    for seed in seeds:
        print(f"\n--- Setting up environment for seed {seed} ---")
        env, env_config = setup_2d_env(env_config_dir, device=args.device)

        video_dir = os.path.join(video_base_dir, f"seed_{seed}") if video_base_dir else None

        result = run_eval_rollouts(
            policy=policy,
            env=env,
            n_episodes=args.n_rollouts,
            exe_steps=exe_steps,
            seed=seed,
            save_videos=args.save_videos,
            video_dir=video_dir,
            video_episodes=args.video_episodes,
            video_fps=args.video_fps,
        )
        all_results.append(result)

    # ------------------------------------------------------------------
    #  Aggregate & save results
    # ------------------------------------------------------------------
    all_success_rates = [r["success_rate"] for r in all_results]
    mean_sr = float(np.mean(all_success_rates))
    std_sr = float(np.std(all_success_rates))

    all_fracs = [r["mean_success_frac"] for r in all_results]
    mean_frac = float(np.mean(all_fracs))
    std_frac = float(np.std(all_fracs))

    all_successes = []
    for r in all_results:
        all_successes.extend(r["successes"])
    overall_sr = float(np.mean(all_successes))

    print(f"\n{'='*60}")
    print(f"Results ({len(all_successes)} total rollouts)")
    print(f"{'='*60}")
    for r in all_results:
        print(
            f"  Seed {r['seed']}: SR={r['success_rate']*100:.1f}%  "
            f"Frac={r['mean_success_frac']:.3f}  "
            f"AvgDist={r['avg_obj_dist']:.4f}  MaxDist={r['max_obj_dist']:.4f}"
        )
    print(f"  Mean SR: {mean_sr*100:.1f}% +/- {std_sr*100:.1f}%")
    print(f"  Mean Frac: {mean_frac:.3f} +/- {std_frac:.3f}")
    print(f"  Overall SR: {overall_sr*100:.1f}%")
    print(f"{'='*60}")

    results = {
        "config": args.config,
        "mode": args.mode,
        "loadpath": args.loadpath,
        "ckpt_path": args.ckpt_path,
        "ckpt_label": ckpt_label,
        "env_config_dir": env_config_dir,
        "exe_steps": exe_steps,
        "n_rollouts_per_seed": args.n_rollouts,
        "seeds": seeds,
        "timestamp": datetime.now().isoformat(),
        "per_seed_results": all_results,
        "mean_success_rate": mean_sr,
        "std_success_rate": std_sr,
        "mean_success_frac": mean_frac,
        "std_success_frac": std_frac,
        "overall_success_rate": overall_sr,
        "total_rollouts": len(all_successes),
    }

    seeds_str = "_".join(map(str, seeds))
    output_file = os.path.join(output_dir, f"eval_{ckpt_label}_seeds{seeds_str}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda x: x if not isinstance(x, np.ndarray) else x.tolist())

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
