"""
Standalone mimicgen multitask evaluation rollouts.

Loads a trained multitask diffusion checkpoint (NO training) and runs N rollouts
per task for one or more seeds, saving a per-episode .mp4 for every rollout plus a
JSON success-rate summary.

This reuses the SAME construction as scripts/train.py and the same per-task eval
wiring (train.py "mimicgen eval wiring" block) so behaviour matches in-training eval,
except: no training loop, configurable n_episodes / seeds, videos written to disk.

Launch it with the SAME --config and architecture flags you trained with, from the
inner `diffuser/` dir with PYTHONPATH=.:.. (architecture args must match the ckpt).

Eval knobs are passed via environment variables (so we don't touch ArgsParser):

  ECDIFF_EVAL_EPISODES   episodes per (task, seed)         [default 50]
  ECDIFF_EVAL_SEEDS      comma-sep seeds                   [default "42,43,44"]
  ECDIFF_EVAL_VIDEO_ROOT output root for videos+summary    [default <logdir>/eval_rollouts]
  ECDIFF_EVAL_CKPT_DIR   override run dir holding ckpt/    [default args.savepath]
  ECDIFF_EVAL_TASKS      comma-sep subset of task names    [default all]
  ECDIFF_EVAL_OVERLAY_KP draw DLP keypoints on video (0/1) [default 0]

Example (12-task relative checkpoint):

  cd /home/ellina/Desktop/EC-Diffuser/diffuser
  ECDIFF_EVAL_EPISODES=50 ECDIFF_EVAL_SEEDS=42,43,44 \
  ECDIFF_EVAL_CKPT_DIR=/home/ellina/Desktop/EC-Diffuser/diffuser/data/multitask/diffusion/mimicgen_multitask/12C_dlp_adalnpint_relative_H16_T5_seed42 \
  ECDIFF_EVAL_VIDEO_ROOT=/home/ellina/Desktop/EC-Diffuser/diffuser/data/multitask/diffusion/mimicgen_multitask/12C_dlp_adalnpint_relative_H16_T5_seed42/eval_rollouts \
  MUJOCO_GL=egl xvfb-run -a python scripts/eval_mimicgen.py \
      --config config.mimicgen_multitask_dlp <same other train flags>
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import json
import random

import numpy as np
import torch

import diffuser.utils as utils
from diffuser.utils.arrays import set_global_device
from diffuser.utils.args import ArgsParser


# -----------------------------------------------------------------------------#
#                   make lpwm-dev / lpwm-copy importable                        #
#                       (copied from scripts/train.py)                          #
# -----------------------------------------------------------------------------#
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# lpwm-dev (3D voxel) / lpwm-copy (2D image) hold the `utils`/`models`/`voxel_models`
# packages the DLP loader needs. They sit next to the EC-Diffuser repo; walk up a few
# levels and add whichever are found (don't rely on a fixed ../../.. depth).
_search = _SCRIPT_DIR
# NB: this mimicgen checkpoint's DLP is a 2D image DLP (dlp_ctor="models:DLP",
# cfg has image_size, no voxel_grid_whd) which lives in lpwm-copy. Both lpwm-copy
# and lpwm-dev define a top-level `models`, so lpwm-copy MUST take priority on
# sys.path or `from models import DLP` resolves to lpwm-dev's 3D (conv3d) variant.
for _ in range(6):
    _search = os.path.dirname(_search)
    for _sibling in ("lpwm-copy", "lpwm-dev"):
        _p = os.path.join(_search, _sibling)
        if os.path.isdir(_p) and _p not in sys.path:
            sys.path.append(_p)


# -----------------------------------------------------------------------------#
#          DLP loading helpers — copied verbatim from scripts/train.py          #
#   (kept local so we don't import train.py, whose module body starts training) #
# -----------------------------------------------------------------------------#
def build_dlp_3d_from_cfg(cfg, device, DLPClass):
    """Build a 3D (voxel) DLP model from config."""
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
    ).to(device)
    model.eval()
    return model


def build_dlp_2d_from_cfg(cfg, device, DLPClass):
    """Build a 2D (image) DLP model from config."""
    model = DLPClass(
        cdim=cfg["ch"],
        image_size=cfg["image_size"],
        normalize_rgb=cfg.get("normalize_rgb", False),
        n_kp_per_patch=cfg["n_kp_per_patch"],
        patch_size=cfg["patch_size"],
        anchor_s=cfg["anchor_s"],
        n_kp_enc=cfg["n_kp_enc"],
        n_kp_prior=cfg["n_kp_prior"],
        pad_mode=cfg["pad_mode"],
        dropout=cfg.get("dropout", 0.0),
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
    ).to(device)
    model.eval()
    return model


def load_dlp_lpwm(dlp_cfg_path, dlp_ckpt_path, device, dlp_ctor="voxel_models:DLP"):
    """Load a DLP model (3D or 2D) based on dlp_ctor (copied from scripts/train.py)."""
    from utils.util_func import get_config

    dev = torch.device(device)
    cfg = get_config(dlp_cfg_path)

    is_2d = "voxel" not in dlp_ctor.lower()
    if is_2d:
        from models import DLP as DLPClass
        model = build_dlp_2d_from_cfg(cfg, dev, DLPClass)
        model.load_state_dict(
            torch.load(dlp_ckpt_path, map_location=dev, weights_only=False)
        )
    else:
        from utils.log_utils import load_checkpoint
        from voxel_models import DLP as DLPClass
        model = build_dlp_3d_from_cfg(cfg, dev, DLPClass)
        _ = load_checkpoint(dlp_ckpt_path, model, None, None, map_location=dev)

    model.eval()
    return model, cfg


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------#
#                                    main                                       #
# -----------------------------------------------------------------------------#
def main():
    args = ArgsParser().parse_args("diffusion")
    set_global_device(args.device)

    if getattr(args, "eval_backend", "none") != "mimicgen":
        print(f"[eval_mimicgen] WARNING: args.eval_backend={getattr(args,'eval_backend',None)} "
              f"(expected 'mimicgen'); continuing anyway.", flush=True)

    # ---- eval knobs (env vars; don't touch ArgsParser) ----
    n_episodes = int(os.environ.get("ECDIFF_EVAL_EPISODES", "50"))
    seeds = [int(s) for s in os.environ.get("ECDIFF_EVAL_SEEDS", "42,43,44").split(",") if s.strip()]
    overlay_kp = os.environ.get("ECDIFF_EVAL_OVERLAY_KP", "0") not in ("0", "", "false", "False")
    only = os.environ.get("ECDIFF_EVAL_TASKS", "").strip()
    only_set = set(t.strip() for t in only.split(",") if t.strip()) if only else None

    # ------------------------------------------------------------------ #
    #                           dataset / renderer                        #
    #            (mirrors scripts/train.py setup, no training)            #
    # ------------------------------------------------------------------ #
    dataset_config = utils.Config(
        args.loader,
        savepath=None,
        dataset_path=args.dataset_path,
        dataset_name=args.dataset,
        horizon=args.horizon,
        obs_only=args.obs_only,
        action_only=args.action_only,
        normalizer=args.normalizer,
        particle_normalizer=args.particle_normalizer,
        preprocess_fns=args.preprocess_fns,
        use_padding=args.use_padding,
        max_path_length=args.max_path_length,
        overfit=args.overfit,
        single_view=(args.input_type == "dlp" and not args.multiview),
        action_z_scale=getattr(args, "action_z_scale", 1.0),
        use_gripper_obs=getattr(args, "use_gripper_obs", False),
        use_bg_obs=getattr(args, "use_bg_obs", False),
        task_entries=getattr(args, "task_entries", None),
        max_demos_per_task=getattr(args, "max_demos_per_task", None),
    )
    render_config = utils.Config(
        args.renderer,
        savepath=None,
        env=None,
        particle_dim=args.features_dim,
        single_view=(args.input_type == "dlp" and not args.multiview),
    )

    dataset = dataset_config()
    renderer = render_config()

    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim
    gripper_dim = getattr(dataset, "gripper_dim", 0)
    bg_dim = getattr(dataset, "bg_dim", 0)

    # ------------------------------------------------------------------ #
    #                          model / diffusion / trainer                #
    # ------------------------------------------------------------------ #
    model_config = utils.Config(
        args.model,
        savepath=None,
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
        n_tasks=getattr(args, "n_tasks", 1),
        split_action_tokens=getattr(args, "split_action_tokens", None),
        action_token_groups=getattr(args, "action_token_groups", None),
        proprio_token_groups=getattr(args, "proprio_token_groups", None),
        aux_action_token_groups=getattr(args, "aux_action_token_groups", None),
        aux_proprio_token_groups=getattr(args, "aux_proprio_token_groups", None),
    )
    diffusion_config = utils.Config(
        args.diffusion,
        savepath=None,
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
        aux_action_loss_weight=getattr(args, "aux_action_loss_weight", 1.0),
    )
    trainer_config = utils.Config(
        utils.Trainer,
        savepath=None,
        train_batch_size=args.batch_size,
        train_lr=args.learning_rate,
        gradient_accumulate_every=args.gradient_accumulate_every,
        ema_decay=args.ema_decay,
        sample_freq=args.sample_freq,
        save_freq=args.save_freq,
        label_freq=int(args.n_train_steps // args.n_saves),
        save_parallel=args.save_parallel,
        results_folder=args.savepath,
        bucket=args.bucket,
        n_reference=args.n_reference,
    )

    model = model_config()
    diffusion = diffusion_config(model)
    trainer = trainer_config(diffusion, dataset, renderer)

    # Point the trainer at the run dir holding ckpt/ (override or args.savepath).
    ckpt_dir = os.environ.get("ECDIFF_EVAL_CKPT_DIR", "").strip() or args.savepath
    trainer.logdir = ckpt_dir
    loaded = trainer.load_latest()
    if not loaded:
        raise RuntimeError(f"No checkpoint found under {os.path.join(ckpt_dir, 'ckpt')}")
    print(f"[eval_mimicgen] loaded checkpoint at step {trainer.step} from {ckpt_dir}", flush=True)

    video_root = os.environ.get("ECDIFF_EVAL_VIDEO_ROOT", "").strip() \
        or os.path.join(ckpt_dir, "eval_rollouts")
    os.makedirs(video_root, exist_ok=True)

    # ------------------------------------------------------------------ #
    #         build per-task eval contexts (mirrors train.py block)       #
    # ------------------------------------------------------------------ #
    from diffuser.envs.mimicgen_dlp_wrapper import DatasetGoalProvider
    from diffuser.eval_utils import setup_mimicgen_env, extract_mimicgen_task_name

    use_absolute_actions = getattr(args, "use_absolute_actions", True)
    _dlp_ctor_eval = getattr(args, "dlp_ctor", "voxel_models:DLP")
    print(f"[eval_mimicgen] use_absolute_actions={use_absolute_actions} "
          f"dlp_ctor={_dlp_ctor_eval}", flush=True)

    if getattr(args, "multitask", False):
        _task_entries = list(getattr(args, "task_entries", []) or [])
    else:
        _task_entries = [{
            "name": getattr(args, "dataset", "task"),
            "task_id": None,
            "calib_h5": getattr(args, "calib_h5_path", None),
            "dlp_ckpt": getattr(args, "dlp_ckpt", None),
            "dlp_cfg": getattr(args, "dlp_cfg", None),
            "pkl": getattr(args, "dataset_path", None),
            "max_steps": getattr(args, "mimicgen_max_steps", 600),
        }]

    if only_set is not None:
        _task_entries = [e for e in _task_entries if e["name"] in only_set]
        if not _task_entries:
            raise RuntimeError(f"ECDIFF_EVAL_TASKS={only} matched no task_entries")

    print(f"[eval_mimicgen] evaluating {len(_task_entries)} task(s): "
          f"{[e['name'] for e in _task_entries]}", flush=True)

    eval_contexts = []
    for _e in _task_entries:
        for _k in ("calib_h5", "dlp_ckpt", "dlp_cfg", "pkl"):
            if _e.get(_k) is None:
                raise RuntimeError(f"mimicgen eval: task '{_e['name']}' missing '{_k}'")
        print(f"[eval_mimicgen] loading DLP for '{_e['name']}': cfg={_e['dlp_cfg']}", flush=True)
        _task_dlp, _ = load_dlp_lpwm(_e["dlp_cfg"], _e["dlp_ckpt"], args.device, dlp_ctor=_dlp_ctor_eval)
        _goal_prov = DatasetGoalProvider(_e["pkl"], shuffle=True)
        _mg_task = getattr(args, "mimicgen_task", None) or extract_mimicgen_task_name(_e["calib_h5"])

        def _make_env_fn(_calib=_e["calib_h5"]):
            def _fn():
                args.calib_h5_path = _calib  # setup_mimicgen_env reads task metadata from this h5
                return setup_mimicgen_env(args, use_absolute_actions=use_absolute_actions)
            return _fn

        eval_contexts.append({
            "name": _e["name"],
            "task_id": (int(_e["task_id"]) if _e.get("task_id") is not None else None),
            "calib_h5": _e["calib_h5"],
            "dlp_model": _task_dlp,
            "goal_provider": _goal_prov,
            "mimicgen_task": _mg_task,
            "make_env_fn": _make_env_fn(),
            "max_steps": int(_e.get("max_steps", getattr(args, "mimicgen_max_steps", 600))),
        })

    mimicgen_cams = tuple(getattr(args, "mimicgen_cams", ("agentview", "sideview")))
    bounds_xyz = getattr(args, "mimicgen_bounds_xyz", ((-2, 2), (-2, 2), (-0.2, 2.5)))
    grid_dhw = getattr(args, "mimicgen_grid_dhw", (128, 128, 128))
    pixel_stride = getattr(args, "mimicgen_pixel_stride", 2)
    random_init = getattr(args, "random_init_eval", False)
    exe_steps = getattr(args, "exe_steps", 1)

    # ------------------------------------------------------------------ #
    #                      seed x task rollout loop                       #
    # ------------------------------------------------------------------ #
    summary = {}
    summary_path = os.path.join(video_root, "summary.json")
    print(f"[eval_mimicgen] n_episodes={n_episodes} seeds={seeds} "
          f"-> {len(seeds) * len(eval_contexts) * n_episodes} total rollouts", flush=True)

    for seed in seeds:
        _seed_everything(seed)
        for ctx in eval_contexts:
            vdir = os.path.join(video_root, ctx["name"], f"seed_{seed}")
            os.makedirs(vdir, exist_ok=True)
            renderer.latent_rep_model = ctx["dlp_model"]
            print(f"\n[eval_mimicgen] >>> task='{ctx['name']}' seed={seed} "
                  f"(task_id={ctx['task_id']}, max_steps={ctx['max_steps']}) -> {vdir}", flush=True)
            try:
                stats = trainer.eval_mimicgen_rollouts(
                    make_env_fn=ctx["make_env_fn"],
                    dlp_model=ctx["dlp_model"],
                    calib_h5_path=ctx["calib_h5"],
                    n_episodes=n_episodes,
                    max_steps=ctx["max_steps"],
                    bounds_xyz=bounds_xyz,
                    grid_dhw=grid_dhw,
                    cams=mimicgen_cams,
                    pixel_stride=pixel_stride,
                    goal_provider=ctx["goal_provider"],
                    random_init=random_init,
                    task=ctx["mimicgen_task"],
                    task_id=ctx["task_id"],
                    renderer_3d=renderer,
                    exe_steps=exe_steps,
                    save_videos=True,
                    video_dir=vdir,
                    video_cams=mimicgen_cams,
                    overlay_keypoints=overlay_kp,
                    log_imagined_states=False,
                    wandb_run=None,
                    video_tag=f"eval/{ctx['name']}",
                )
            except Exception as err:
                import traceback
                print(f"[eval_mimicgen] task={ctx['name']} seed={seed} FAILED, skipping: {err}", flush=True)
                traceback.print_exc()
                summary[f"{ctx['name']}/seed_{seed}"] = {"error": str(err)}
                with open(summary_path, "w") as f:
                    json.dump(summary, f, indent=2)
                continue

            summary[f"{ctx['name']}/seed_{seed}"] = stats
            print(f"[eval_mimicgen] task={ctx['name']} seed={seed} :: {stats}", flush=True)
            # incremental write so progress survives a crash
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)

    # aggregate per-task mean success rate across seeds
    per_task = {}
    for ctx in eval_contexts:
        srs = [summary[f"{ctx['name']}/seed_{s}"].get("sim/success_rate")
               for s in seeds
               if f"{ctx['name']}/seed_{s}" in summary
               and "sim/success_rate" in summary[f"{ctx['name']}/seed_{s}"]]
        if srs:
            per_task[ctx["name"]] = float(np.mean(srs))
    summary["_per_task_mean_success_rate"] = per_task
    if per_task:
        summary["_overall_mean_success_rate"] = float(np.mean(list(per_task.values())))
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[eval_mimicgen] DONE. videos+summary under {video_root}", flush=True)
    print(f"[eval_mimicgen] per-task mean success rate: {per_task}", flush=True)


if __name__ == "__main__":
    main()
