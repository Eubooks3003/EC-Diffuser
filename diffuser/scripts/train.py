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
# lpwm-dev and lpwm-copy live near it, but how near depends on the checkout:
# they may be siblings of EC-Diffuser (../../..) or of its parent (../../../..).
# The old code only tried the latter, so on a sibling-of-EC-Diffuser layout the
# DLP import died with "No module named 'utils'".
_SCRIPT_DIR = os.path.dirname(__file__)
_LPWM_UPS = ("..", "../..", "../../..", "../../../..")


def _find_lpwm_root(name):
    """Locate an lpwm checkout: $LPWM_COPY/$LPWM_DEV wins, else search upward."""
    env = os.environ.get(name.replace("-", "_").upper())  # LPWM_COPY / LPWM_DEV
    if env and os.path.isdir(env):
        return os.path.abspath(env)
    for _up in _LPWM_UPS:
        _p = os.path.abspath(os.path.join(_SCRIPT_DIR, _up, name))
        if os.path.isdir(_p):
            return _p
    return None


def ensure_lpwm_on_path(dlp_ctor):
    """Put the correct lpwm checkout FIRST on sys.path for this dlp_ctor.

    Both checkouts ship a top-level models.py defining DLP, and a `utils`
    namespace package that merges across them. Order therefore decides which
    model class you get, silently:
      - "models:DLP" (2D)       -> lpwm-copy must win
      - "voxel_models:DLP" (3D) -> lpwm-dev must win; voxel_models needs
        lpwm-dev's utils.loss_functions (lpwm-copy's lacks bce_logits_weighted)
    """
    prefer_2d = "voxel" not in str(dlp_ctor).lower()
    order = ("lpwm-copy", "lpwm-dev") if prefer_2d else ("lpwm-dev", "lpwm-copy")
    roots = [r for r in (_find_lpwm_root(n) for n in order) if r]
    if not roots:
        raise RuntimeError(
            f"Could not locate lpwm-copy / lpwm-dev near {_SCRIPT_DIR}. "
            f"Set LPWM_COPY (and LPWM_DEV) to their absolute paths.")
    for _p in reversed(roots):          # insert in reverse -> roots[0] ends up first
        if _p in sys.path:
            sys.path.remove(_p)
        sys.path.insert(0, _p)
    print(f"[lpwm] sys.path order for dlp_ctor={dlp_ctor}: {roots}", flush=True)


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
    ensure_lpwm_on_path(dlp_ctor)
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
    aux_action_token_groups=getattr(args, 'aux_action_token_groups', None),
    self_cond_action=getattr(args, 'self_cond_action', False),
    aux_proprio_token_groups=getattr(args, 'aux_proprio_token_groups', None),
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
    aux_action_loss_weight=getattr(args, 'aux_action_loss_weight', 1.0),
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

        # Per-task cameras: pick_place_d0's arena (BinsArena) has no 'sideview',
        # so it was preprocessed with ('agentview', 'frontview'). Read the
        # cameras the DLP was actually trained on from the task pkl's meta,
        # exactly as scripts/eval_paper.py does — requesting the wrong second
        # camera raises KeyError('sideview_depth') during env calibration.
        _task_cams = list(getattr(args, "mimicgen_cams", ["agentview", "sideview"]))
        try:
            import pickle as _pickle
            with open(_e["pkl"], "rb") as _fh:
                _meta = _pickle.load(_fh).get("meta", {})
            _meta_cams = _meta.get("cameras", None)
            if _meta_cams and list(_meta_cams) != _task_cams:
                print(f"[mimicgen eval] '{_e['name']}' cam override: {_task_cams} -> "
                      f"{list(_meta_cams)} (from pkl meta)", flush=True)
                _task_cams = list(_meta_cams)
        except Exception as _cam_err:
            print(f"[mimicgen eval] WARNING: could not read meta cameras for "
                  f"'{_e['name']}': {_cam_err}", flush=True)

        def _make_env_fn(_calib=_e["calib_h5"], _cams=_task_cams):
            def _fn():
                args.calib_h5_path = _calib  # setup_mimicgen_env reads task metadata from this h5
                args.mimicgen_cams = _cams   # and reads the camera list off args
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
            "cams": _task_cams,
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
                        cams=ctx["cams"],
                        pixel_stride=getattr(args, "mimicgen_pixel_stride", 2),
                        goal_from_env_fn=getattr(args, "goal_from_env_fn", None),
                        goal_provider=ctx["goal_provider"],
                        random_init=getattr(args, "random_init_eval", False),
                        task=ctx["mimicgen_task"],
                        renderer_3d=renderer,
                        exe_steps=getattr(args, "exe_steps", 1),
                        task_id=ctx["task_id"],
                        video_cams=ctx["cams"],
                        overlay_keypoints=True,        # NEW: kps on the rollout video
                        wandb_run=wandb_run,           # NEW: log video to wandb
                        video_tag=f"eval/{ctx['name']}",
                        # Per-task subdir: without this every task writes
                        # ep_000.mp4 into the same step_<N>/ dir and each one
                        # overwrites the last, leaving only the final task.
                        video_dir=os.path.join(args.savepath, "eval_videos",
                                               f"step_{trainer.step}", ctx["name"]),
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
