"""
Standalone RLBench eval for the language-conditioned 2D-DLP policy.

Loads a checkpoint (or the latest by default) from an existing training run
and runs rollouts without starting a training loop. Useful for iterating on
eval-side bugs (planning errors, action conversion, video saving) without
killing ongoing training.

Usage:
    xvfb-run -a python diffuser/scripts/eval_rlbench.py \
        --config config.rlbench_close_jar_dlp \
        --dataset close_jar \
        --num_entity 16 --input_type dlp --seed 42 \
        --ckpt_step latest \
        --n_episodes 5 \
        --max_steps 400
"""
import argparse
import os
import sys

import numpy as np
import torch

import diffuser.utils as utils
from diffuser.utils.arrays import apply_dict


def _build_args(raw_argv):
    """Rebuild the same args object train.py uses, plus our own overrides."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--ckpt_step", default="latest",
                     help="Checkpoint step to load: 'latest' or an int (e.g. 60000)")
    pre.add_argument("--n_episodes", type=int, default=5)
    pre.add_argument("--max_steps", type=int, default=400)
    pre.add_argument("--eval_prefix", default="standalone_eval",
                     help="Subdir name under <savepath>/eval_videos/ for outputs")
    ours, rest = pre.parse_known_args(raw_argv)

    # Hand the rest of argv to train-time Parser (which reads the config file).
    from diffuser.utils.args import ArgsParser
    sys.argv = [sys.argv[0]] + rest
    args = ArgsParser().parse_args("diffusion")
    # attach our extras
    args._ckpt_step = ours.ckpt_step
    args._n_episodes = ours.n_episodes
    args._max_steps = ours.max_steps
    args._eval_prefix = ours.eval_prefix
    return args


def main(argv):
    args = _build_args(argv)
    print(f"[eval_rlbench] savepath = {args.savepath}", flush=True)

    # -------------------------------------------------------------------
    # Dataset (for normalizer) -- mirrors train.py
    # -------------------------------------------------------------------
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
    print(f"[eval_rlbench] obs_dim={observation_dim} action_dim={action_dim} "
          f"gripper_dim={gripper_dim} bg_dim={bg_dim}", flush=True)

    # -------------------------------------------------------------------
    # Model + Diffusion
    # -------------------------------------------------------------------
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

    # -------------------------------------------------------------------
    # Load checkpoint
    # -------------------------------------------------------------------
    ckpt_dir = os.path.join(args.savepath, "ckpt")
    if args._ckpt_step == "latest":
        files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
        if not files:
            raise RuntimeError(f"No .pt checkpoint under {ckpt_dir}")

        def _step_of(f):
            try:
                return int(f.rsplit("_step", 1)[-1].split(".")[0])
            except Exception:
                return -1
        files.sort(key=_step_of)
        ckpt_path = os.path.join(ckpt_dir, files[-1])
    else:
        matches = [f for f in os.listdir(ckpt_dir) if f.endswith(f"_step{args._ckpt_step}.pt")]
        if not matches:
            raise RuntimeError(f"No ckpt for step {args._ckpt_step} in {ckpt_dir}")
        ckpt_path = os.path.join(ckpt_dir, matches[0])
    print(f"[eval_rlbench] loading checkpoint: {ckpt_path}", flush=True)
    sd = torch.load(ckpt_path, map_location=args.device)
    state = sd.get("ema", sd.get("model", sd))
    # The saved dict may be the full EMA diffusion state
    diffusion.load_state_dict(state, strict=False)

    # -------------------------------------------------------------------
    # DLP encoder for live encoding
    # -------------------------------------------------------------------
    dlp_ckpt = args.dlp_ckpt
    dlp_cfg_path = args.dlp_cfg
    _dlp_ctor_eval = getattr(args, "dlp_ctor", "models:DLP")
    print(f"[eval_rlbench] loading DLP from {dlp_ckpt}", flush=True)

    # Inline load_dlp_lpwm + build_dlp_2d_from_cfg (copied from train.py so we
    # don't execute that file's module-level training setup on import).
    def _build_dlp_2d_from_cfg(cfg, dev, DLPClass):
        model = DLPClass(
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
        model.eval()
        return model

    from utils.util_func import get_config
    dev = torch.device(args.device)
    dlp_cfg = get_config(dlp_cfg_path)
    is_2d = "voxel" not in _dlp_ctor_eval.lower()
    if is_2d:
        from models import DLP as _DLPClass
        dlp_model = _build_dlp_2d_from_cfg(dlp_cfg, dev, _DLPClass)
        dlp_model.load_state_dict(
            torch.load(dlp_ckpt, map_location=dev, weights_only=False)
        )
    else:
        from utils.log_utils import load_checkpoint
        from voxel_models import DLP as _DLPClass
        # 3D path shouldn't occur for RLBench 2D setup, but kept for symmetry
        raise RuntimeError(f"3D DLP path not supported in eval_rlbench; dlp_ctor={_dlp_ctor_eval}")
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
        # Match preprocessing's per-view batching (see
        # lpwm-copy/scripts/ec_diffuser_rlbench_multiview_preprocess.py:332-360).
        # Encoding all views in a single batch vs one-at-a-time produces
        # different tokens because of cross-batch dependencies in the encoder.
        rgbs = torch.from_numpy(np.asarray(rgbs_np)).float() / 255.0
        rgbs = rgbs.permute(0, 3, 1, 2).to(args.device)  # (V, 3, H, W)
        if rgbs.shape[-1] != _img_size:
            rgbs = torch.nn.functional.interpolate(
                rgbs, size=(_img_size, _img_size), mode="bilinear", align_corners=False
            )
        per_view_toks = []
        per_view_bg = []
        for vi in range(rgbs.shape[0]):
            chunk = rgbs[vi:vi + 1]  # (1, 3, H, W)
            enc = dlp_model.encode_all(chunk, deterministic=True)
            toks_v, bg_v = _pack_tokens_2d(enc)
            per_view_toks.append(toks_v)
            per_view_bg.append(bg_v)
        toks = torch.cat(per_view_toks, dim=0)  # (V, K, Dtok)
        bg = torch.cat(per_view_bg, dim=0)      # (V, bg_F)
        V, K, Dtok = toks.shape
        return {
            "tokens": toks.reshape(V * K, Dtok).cpu().numpy(),
            "bg": bg.reshape(-1).cpu().numpy(),
        }

    # -------------------------------------------------------------------
    # Build a minimal trainer shim so we can call eval_rlbench_rollouts
    # -------------------------------------------------------------------
    from diffuser.utils.training import Trainer
    trainer_config = utils.Config(
        Trainer,
        savepath=(args.savepath, f"trainer_eval_{args._ckpt_step}.pkl"),
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

    # Copy the loaded weights into ema_model (which eval uses).
    trainer.ema_model.load_state_dict(state, strict=False)
    # Use the file's step as trainer.step so video path is tagged correctly.
    try:
        trainer.step = int(os.path.basename(ckpt_path).rsplit("_step", 1)[-1].split(".")[0])
    except Exception:
        trainer.step = 0

    # -------------------------------------------------------------------
    # Env factory + policy factory
    # -------------------------------------------------------------------
    def make_env_fn():
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
        # Expose the DLP encoder+decoder to the trainer so it can optionally
        # render imagined particle states (ECDIFF_SAVE_IMAGINED_RECON=1).
        _env._dlp_model = dlp_model
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

    # -------------------------------------------------------------------
    # Run
    # -------------------------------------------------------------------
    # Auto-populate diagnostic flags from config so `python eval_rlbench.py ...`
    # works without any exported env vars. setdefault() lets shell overrides win.
    if getattr(args, "save_gt_video", False):
        os.environ.setdefault("ECDIFF_SAVE_GT_VIDEO", "1")
    _demo_root_cfg = getattr(args, "demo_dataset_root", None)
    if _demo_root_cfg:
        os.environ.setdefault("ECDIFF_DEMO_ROOT", str(_demo_root_cfg))
    if getattr(args, "save_imagined", False):
        os.environ.setdefault("ECDIFF_SAVE_IMAGINED", "1")
    if getattr(args, "save_imagined_recon", False):
        os.environ.setdefault("ECDIFF_SAVE_IMAGINED_RECON", "1")

    sim_stats = trainer.eval_rlbench_rollouts(
        make_env_fn=make_env_fn,
        make_policy_fn=make_policy_fn,
        n_episodes=args._n_episodes,
        max_steps=args._max_steps,
        task_name=args.dataset,
        exe_steps=getattr(args, "exe_steps", 1),
    )
    print(f"\n[eval_rlbench] final :: {sim_stats}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
