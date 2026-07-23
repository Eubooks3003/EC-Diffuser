import warnings
warnings.filterwarnings("ignore")

import os
import sys

import wandb
import torch

import diffuser.utils as utils
from diffuser.utils.arrays import set_global_device
from diffuser.utils.args import ArgsParser

import logging
logging.basicConfig(level=logging.WARNING, force=True) 


# -----------------------------------------------------------------------------#
#                   make lpwm-dev / lpwm-copy importable                        #
# -----------------------------------------------------------------------------#

# train.py is usually at:  EC-Diffuser/diffuser/scripts/train.py
# lpwm-dev and lpwm-copy are siblings of EC-Diffuser
_SCRIPT_DIR = os.path.dirname(__file__)
for _sibling in ("lpwm-dev", "lpwm-copy"):
    _p = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "..", "..", _sibling))
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)


# -----------------------------------------------------------------------------#
#                          LPWM DLP loading (cfg + ckpt)                        #
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
        separate_depth_features=cfg.get("separate_depth_features", False),
        depth_feature_dim=cfg.get("depth_feature_dim", 0),
        split_loss=cfg.get("split_loss", False),
        depth_loss_ratio=cfg.get("depth_loss_ratio", 1.0),
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


def load_dlp_lpwm(dlp_cfg_path: str, dlp_ckpt_path: str, device: str,
                   dlp_ctor: str = "voxel_models:DLP"):
    """
    Load a DLP model (3D or 2D) based on dlp_ctor.

    dlp_ctor format: "module:ClassName"
      - "voxel_models:DLP"  -> 3D DLP from lpwm-dev
      - "models:DLP"        -> 2D DLP from lpwm-copy
    """
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


# -----------------------------------------------------------------------------#
#                                   setup                                      #
# -----------------------------------------------------------------------------#

args = ArgsParser().parse_args("diffusion")
set_global_device(args.device)

eval_backend = getattr(args, "eval_backend", "none")   # "none" | "mimicgen" | "isaac"
eval_freq = int(getattr(args, "eval_freq", 0) or 0)
do_eval = (eval_freq > 0) and (eval_backend != "none")

print(f"[eval cfg] eval_backend={eval_backend} eval_freq={eval_freq} do_eval={do_eval}", flush=True)


# -----------------------------------------------------------------------------#
#                                   dataset                                    #
# -----------------------------------------------------------------------------#

dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, "dataset_config.pkl"),
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
    action_z_scale=getattr(args, 'action_z_scale', 1.0),
    use_gripper_obs=getattr(args, 'use_gripper_obs', False),
    use_bg_obs=getattr(args, 'use_bg_obs', False),
    task_entries=getattr(args, 'task_entries', None),
    max_demos_per_task=getattr(args, 'max_demos_per_task', None),
)

render_config = utils.Config(
    args.renderer,
    savepath=(args.savepath, "render_config.pkl"),
    env=None,
    particle_dim=args.features_dim,
    single_view=(args.input_type == "dlp" and not args.multiview),
)

dataset = dataset_config()

print("DATASTE TYPE: ", dataset.__class__.__name__)
renderer = render_config()

# Load DLP for renderer reference renders (independent of eval backend)
dlp_cfg_path = getattr(args, "dlp_cfg", None)
dlp_ckpt_path = getattr(args, "dlp_ckpt", None)
if dlp_cfg_path and dlp_ckpt_path:
    print(f"[renderer] loading DLP for reference renders: cfg={dlp_cfg_path} ckpt={dlp_ckpt_path}", flush=True)
    _dlp_ctor = getattr(args, "dlp_ctor", "voxel_models:DLP")
    _renderer_dlp, _ = load_dlp_lpwm(dlp_cfg_path, dlp_ckpt_path, args.device, dlp_ctor=_dlp_ctor)
    renderer.latent_rep_model = _renderer_dlp
else:
    print("[renderer] no DLP cfg/ckpt provided, reference renders will be skipped", flush=True)

print("renderer: ", renderer)

observation_dim = dataset.observation_dim
action_dim = dataset.action_dim
gripper_dim = getattr(dataset, 'gripper_dim', 0)
bg_dim = getattr(dataset, 'bg_dim', 0)

if gripper_dim > 0:
    print(f"[train] Using gripper observations: gripper_dim={gripper_dim}")
if bg_dim > 0:
    print(f"[train] Using background features: bg_dim={bg_dim}")


# -----------------------------------------------------------------------------#
#                              model & trainer                                 #
# -----------------------------------------------------------------------------#

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
    n_tasks=getattr(args, 'n_tasks', 1),
    split_action_tokens=getattr(args, 'split_action_tokens', None),
    action_token_groups=getattr(args, 'action_token_groups', None),
    proprio_token_groups=getattr(args, 'proprio_token_groups', None),
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
)

trainer_config = utils.Config(
    utils.Trainer,
    savepath=(args.savepath, "trainer_config.pkl"),
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


# -----------------------------------------------------------------------------#
#                         test forward & backward pass                          #
# -----------------------------------------------------------------------------#

print("Testing forward...", end=" ", flush=True)
batch = utils.batchify(dataset[0])
loss, _ = diffusion.loss(*batch)
loss.backward()
print("✓", flush=True)


# -----------------------------------------------------------------------------#
#                                    wandb                                     #
# -----------------------------------------------------------------------------#

wandb_run = wandb.init(
    entity=args.wandb_entity,
    project=args.wandb_project,
    group=args.wandb_group_name,
    config=args,
    sync_tensorboard=False,
    settings=wandb.Settings(start_method="fork"),
)
wandb_run.name = f"{args.dataset}_H{args.horizon}_exe{getattr(args, 'exe_steps', 1)}"


# -----------------------------------------------------------------------------#
#                               mimicgen eval wiring                            #
# -----------------------------------------------------------------------------#

# Multitask in-training eval: build one eval context per task (every eval_freq
# epochs we run 1 rollout for EACH task, with DLP keypoints overlaid on the video).
# Each context caches that task's DLP (loaded once) + goal provider + env factory.
eval_contexts = []
use_absolute_actions = getattr(args, "use_absolute_actions", True)

if do_eval and eval_backend == "mimicgen":
    from diffuser.envs.mimicgen_dlp_wrapper import DatasetGoalProvider
    from diffuser.eval_utils import setup_mimicgen_env, extract_mimicgen_task_name

    _dlp_ctor_eval = getattr(args, "dlp_ctor", "voxel_models:DLP")
    print(f"[mimicgen eval] use_absolute_actions = {use_absolute_actions}", flush=True)

    # Decide which task entries to evaluate.
    if getattr(args, "multitask", False):
        _task_entries = list(getattr(args, "task_entries", []) or [])
        _only = getattr(args, "eval_task", "") or ""   # optional: restrict to one task
        if _only:
            _task_entries = [e for e in _task_entries if e["name"] == _only]
            if not _task_entries:
                raise RuntimeError(f"--eval_task='{_only}' not in task_entries")
    else:
        # single-task config: synthesize one entry from args
        _task_entries = [{
            "name": getattr(args, "dataset", "task"),
            "task_id": None,
            "calib_h5": getattr(args, "calib_h5_path", None),
            "dlp_ckpt": getattr(args, "dlp_ckpt", None),
            "dlp_cfg": getattr(args, "dlp_cfg", None),
            "pkl": getattr(args, "dataset_path", None),
            "max_steps": getattr(args, "mimicgen_max_steps", 600),
        }]

    print(f"[mimicgen eval] evaluating {len(_task_entries)} task(s): "
          f"{[e['name'] for e in _task_entries]}", flush=True)

    for _e in _task_entries:
        for _k in ("calib_h5", "dlp_ckpt", "dlp_cfg", "pkl"):
            if _e.get(_k) is None:
                raise RuntimeError(f"mimicgen eval: task '{_e['name']}' missing '{_k}'")
        print(f"[mimicgen eval] loading DLP for '{_e['name']}': cfg={_e['dlp_cfg']}", flush=True)
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

    # Default renderer DLP = first task's (used for training reference renders too).
    if eval_contexts:
        renderer.latent_rep_model = eval_contexts[0]["dlp_model"]


# -----------------------------------------------------------------------------#
#                                  main loop                                   #
# -----------------------------------------------------------------------------#

n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

# Auto-resume from latest checkpoint if one exists
start_epoch = 0
if trainer.load_latest():
    start_epoch = trainer.step // int(args.n_steps_per_epoch)
    print(f"[resume] Resuming from step {trainer.step}, skipping to epoch {start_epoch}/{n_epochs}", flush=True)

for i in range(start_epoch, n_epochs):
    print(f"Epoch {i} / {n_epochs} | {args.savepath}", flush=True)
    trainer.train(n_train_steps=args.n_steps_per_epoch)

    # eval AFTER each epoch; with eval_freq=1 it runs every epoch
    if do_eval and ((i + 1) % eval_freq == 0):
        # Save checkpoint before eval (synced with eval_freq)
        trainer.save(i)
        print(f"[eval] starting {eval_backend} eval at epoch={i} step={trainer.step}", flush=True)

        if eval_backend == "mimicgen":
            # Run 1 rollout per task, each with DLP keypoints overlaid on the video.
            per_task_sr = {}
            for ctx in eval_contexts:
                print(f"[mimicgen eval] >>> task '{ctx['name']}' "
                      f"(task_id={ctx['task_id']}, max_steps={ctx['max_steps']})", flush=True)
                try:
                    renderer.latent_rep_model = ctx["dlp_model"]
                    sim_stats = trainer.eval_mimicgen_rollouts(
                        make_env_fn=ctx["make_env_fn"],
                        dlp_model=ctx["dlp_model"],
                        calib_h5_path=ctx["calib_h5"],
                        n_episodes=getattr(args, "mimicgen_eval_episodes", 1),
                        max_steps=ctx["max_steps"],
                        bounds_xyz=getattr(args, "mimicgen_bounds_xyz", ((-2, 2), (-2, 2), (-0.2, 2.5))),
                        grid_dhw=getattr(args, "mimicgen_grid_dhw", (128, 128, 128)),
                        cams=getattr(args, "mimicgen_cams", ("agentview", "sideview")),
                        pixel_stride=getattr(args, "mimicgen_pixel_stride", 2),
                        goal_from_env_fn=getattr(args, "goal_from_env_fn", None),
                        goal_provider=ctx["goal_provider"],
                        random_init=getattr(args, "random_init_eval", False),
                        task=ctx["mimicgen_task"],
                        renderer_3d=renderer,
                        exe_steps=getattr(args, "exe_steps", 1),
                        task_id=ctx["task_id"],
                        video_cams=getattr(args, "mimicgen_cams", ("agentview", "sideview")),
                        overlay_keypoints=True,        # NEW: kps on the rollout video
                        wandb_run=wandb_run,           # NEW: log video to wandb
                        video_tag=f"eval/{ctx['name']}",
                    )
                except Exception as _eval_err:
                    # Don't let one task's eval (e.g. an env-setup/render failure)
                    # abort the whole training run — log it and move on.
                    import traceback
                    print(f"[mimicgen eval] task={ctx['name']} FAILED, skipping: "
                          f"{_eval_err}", flush=True)
                    traceback.print_exc()
                    continue
                sr = float(sim_stats.get("sim/success_rate", 0.0))
                per_task_sr[ctx["name"]] = sr
                wandb.log({
                    "step": trainer.step,
                    f"sim/{ctx['name']}/success_rate": sr,
                    f"sim/{ctx['name']}/avg_len": float(sim_stats.get("sim/avg_len", 0.0)),
                })
                print(f"[mimicgen eval] task={ctx['name']} :: {sim_stats}", flush=True)

            if per_task_sr:
                mean_sr = sum(per_task_sr.values()) / len(per_task_sr)
                wandb.log({"step": trainer.step, "sim/mean_success_rate": mean_sr})
                print(f"[mimicgen eval] epoch={i} mean_success_rate={mean_sr:.3f} :: {per_task_sr}", flush=True)

        else:
            raise RuntimeError(
                f"eval_backend={eval_backend} not supported in this script without extra wiring."
            )
