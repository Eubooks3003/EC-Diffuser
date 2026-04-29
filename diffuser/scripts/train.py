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
    # Skip the buffer's legacy single_view hack (which halves bg_features assuming
    # a 2-view pkl) when we're doing our own view slicing via use_views.
    single_view=(
        args.input_type == "dlp"
        and not args.multiview
        and getattr(args, 'use_views', None) is None
    ),
    action_z_scale=getattr(args, 'action_z_scale', 1.0),
    use_gripper_obs=getattr(args, 'use_gripper_obs', False),
    use_bg_obs=getattr(args, 'use_bg_obs', False),
    use_views=getattr(args, 'use_views', None),
    num_source_views=getattr(args, 'num_source_views', None),
    action_normalizer=getattr(args, 'action_normalizer', None),
    keypose_mode=getattr(args, 'keypose_mode', False),
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
    lang_dim=getattr(args, 'lang_dim', 0),
    act_pos_dim=getattr(args, 'act_pos_dim', 3),
    act_rot_dim=getattr(args, 'act_rot_dim', 3),
    act_grip_dim=getattr(args, 'act_grip_dim', 1),
    prop_pos_dim=getattr(args, 'prop_pos_dim', 3),
    prop_rot_dim=getattr(args, 'prop_rot_dim', 6),
    prop_grip_dim=getattr(args, 'prop_grip_dim', 1),
    split_action_tokens=getattr(args, 'split_action_tokens', None),
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

dlp_model = None
dlp_cfg = None
make_env_fn = None
calib_h5_path = None
goal_provider = None  # NEW: for dataset-based goal conditioning

if do_eval and eval_backend == "mimicgen":
    calib_h5_path = getattr(args, "calib_h5_path", None)
    dlp_ckpt = getattr(args, "dlp_ckpt", None)
    dlp_cfg_path = getattr(args, "dlp_cfg", None)

    if calib_h5_path is None:
        raise RuntimeError("eval_backend='mimicgen' requires --calib_h5_path")
    if dlp_ckpt is None:
        raise RuntimeError("eval_backend='mimicgen' requires --dlp_ckpt")
    if dlp_cfg_path is None:
        raise RuntimeError("eval_backend='mimicgen' requires --dlp_cfg (LPWM cfg used to build DLP)")

    print(f"[mimicgen eval] loading DLP from cfg={dlp_cfg_path} ckpt={dlp_ckpt}", flush=True)
    _dlp_ctor_eval = getattr(args, "dlp_ctor", "voxel_models:DLP")
    dlp_model, dlp_cfg = load_dlp_lpwm(dlp_cfg_path, dlp_ckpt, args.device, dlp_ctor=_dlp_ctor_eval)

    renderer.latent_rep_model = dlp_model

    # NEW: Set up goal provider from dataset for init_state + goal pairing
    # This ensures eval uses the same (init_state, goal) pairs as training
    from diffuser.envs.mimicgen_dlp_wrapper import DatasetGoalProvider
    print("=" * 60)
    print("[mimicgen eval] Setting up DatasetGoalProvider for goal conditioning")
    print(f"[mimicgen eval]   Loading from: {args.dataset_path}")
    goal_provider = DatasetGoalProvider(args.dataset_path, shuffle=True)
    print("[mimicgen eval] ✓ Goal provider ready - will use paired (init_state, goal_tokens)")
    print("=" * 60, flush=True)

    # NEW: Check if using absolute actions (control_delta=False)
    use_absolute_actions = getattr(args, "use_absolute_actions", True)
    print(f"[mimicgen eval] use_absolute_actions = {use_absolute_actions}", flush=True)

    # Extract task name from HDF5 for task-specific voxel bounds
    from diffuser.eval_utils import extract_mimicgen_task_name
    mimicgen_task = getattr(args, "mimicgen_task", None)  # Allow config override
    if mimicgen_task is None:
        mimicgen_task = extract_mimicgen_task_name(calib_h5_path)
    if mimicgen_task is not None:
        print(f"[mimicgen eval] Using task-specific bounds for task: '{mimicgen_task}'")
    else:
        print(f"[mimicgen eval] WARNING: Could not determine task name, using default bounds")

    def make_env_fn():
        from diffuser.eval_utils import setup_mimicgen_env
        return setup_mimicgen_env(args, use_absolute_actions=use_absolute_actions)


make_policy_fn = None
if do_eval and eval_backend == "rlbench":
    dlp_ckpt = getattr(args, "dlp_ckpt", None)
    dlp_cfg_path = getattr(args, "dlp_cfg", None)
    if dlp_ckpt is None or dlp_cfg_path is None:
        raise RuntimeError("eval_backend='rlbench' requires --dlp_ckpt and --dlp_cfg")

    print(f"[rlbench eval] loading DLP from cfg={dlp_cfg_path} ckpt={dlp_ckpt}", flush=True)
    _dlp_ctor_eval = getattr(args, "dlp_ctor", "models:DLP")
    dlp_model, dlp_cfg = load_dlp_lpwm(
        dlp_cfg_path, dlp_ckpt, args.device, dlp_ctor=_dlp_ctor_eval
    )
    dlp_model.eval()

    # Build an online 5/4-view DLP encode callback that matches preprocess.pack_tokens_2d.
    _cams = getattr(args, "rlbench_cams",
                    ["front", "overhead", "left_shoulder", "right_shoulder"])
    _img_size = int(getattr(args, "rlbench_image_size", 128))

    def _pack_tokens_2d(enc):
        import torch as _torch
        z = enc["z"][:, 0]
        z_scale = enc["z_scale"][:, 0]
        z_depth = enc["z_depth"][:, 0]
        z_feat = enc["z_features"][:, 0]
        z_bg = enc["z_bg_features"][:, 0]
        obj_on = enc.get("z_obj_on", enc.get("obj_on"))[:, 0]
        if obj_on.dim() == 2:
            obj_on = obj_on.unsqueeze(-1)
        if z_bg.dim() == 3:
            z_bg = z_bg.squeeze(1)
        return _torch.cat([z, z_scale, z_depth, obj_on, z_feat], dim=-1), z_bg

    @torch.no_grad()
    def _dlp_encode_fn(rgbs_np):
        """rgbs_np: (V, H, W, 3) uint8 -> dict(tokens=(V*K, Dtok), bg=(V*bg_F,))."""
        import numpy as _np
        import torch as _torch
        rgbs = _torch.from_numpy(_np.asarray(rgbs_np)).float() / 255.0
        rgbs = rgbs.permute(0, 3, 1, 2).to(args.device)  # (V, 3, H, W)
        if rgbs.shape[-1] != _img_size:
            rgbs = _torch.nn.functional.interpolate(
                rgbs, size=(_img_size, _img_size), mode="bilinear", align_corners=False
            )
        # Match preprocessing's per-view batching (preprocess script encodes
        # one view at a time). All-views-in-one-batch gives different tokens
        # due to cross-batch dependency in the encoder.
        per_view_toks = []
        per_view_bg = []
        for vi in range(rgbs.shape[0]):
            chunk = rgbs[vi:vi + 1]
            enc = dlp_model.encode_all(chunk, deterministic=True)
            toks_v, bg_v = _pack_tokens_2d(enc)
            per_view_toks.append(toks_v)
            per_view_bg.append(bg_v)
        toks = _torch.cat(per_view_toks, dim=0)
        bg = _torch.cat(per_view_bg, dim=0)
        V, K, Dtok = toks.shape
        return {
            "tokens": toks.reshape(V * K, Dtok).cpu().numpy(),
            "bg": bg.reshape(-1).cpu().numpy(),
        }

    # Cached across eval cycles. CoppeliaSim's headless OpenGL has a
    # cross-launch texture-loss bug: every launch after the first in a
    # given process renders with a flat washed-yellow floor instead of the
    # textured scene. Reusing one env across all eval cycles avoids it.
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
            episode_length=int(getattr(args, "rlbench_max_steps", 400)),
            delta_actions=not bool(getattr(args, "use_absolute_actions", True)),
        )
        # Expose DLP decoder for imagined-recon video rendering during eval.
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
            sim_stats = trainer.eval_mimicgen_rollouts(
                make_env_fn=make_env_fn,
                dlp_model=dlp_model,
                calib_h5_path=calib_h5_path,
                n_episodes=getattr(args, "mimicgen_eval_episodes", 5),
                max_steps=getattr(args, "mimicgen_max_steps", 50),
                bounds_xyz=getattr(args, "mimicgen_bounds_xyz", ((-2, 2), (-2, 2), (-0.2, 2.5))),
                grid_dhw=getattr(args, "mimicgen_grid_dhw", (128, 128, 128)),
                cams=getattr(args, "mimicgen_cams", ("agentview", "sideview")),
                pixel_stride=getattr(args, "mimicgen_pixel_stride", 2),
                goal_from_env_fn=getattr(args, "goal_from_env_fn", None),
                goal_provider=goal_provider,  # NEW: dataset-based goal provider
                random_init=getattr(args, "random_init_eval", False),  # NEW: random vs dataset init
                task=mimicgen_task,  # Task name for task-specific voxel bounds
                renderer_3d=renderer,
                exe_steps=getattr(args, "exe_steps", 1),  # ACTION CHUNKING: how many actions to execute per plan
            )

            # avoid double-prefix if eval returns sim/... already
            log_stats = {k if k.startswith("sim/") else f"sim/{k}": v for k, v in sim_stats.items()}
            wandb.log({"step": trainer.step, **log_stats})
            print(f"[mimicgen eval] epoch={i} :: {sim_stats}", flush=True)

        elif eval_backend == "rlbench":
            # Auto-populate diagnostic flags from config so the user can run
            # plain `python train.py ...` without exporting ECDIFF_* vars.
            # setdefault() lets shell-exported vars still win if set.
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
                n_episodes=getattr(args, "rlbench_eval_episodes", 5),
                max_steps=getattr(args, "rlbench_max_steps", 400),
                task_name=getattr(args, "dataset", None),
                exe_steps=getattr(args, "exe_steps", 1),
            )
            log_stats = {k if k.startswith("sim/") else f"sim/{k}": v for k, v in sim_stats.items()}
            wandb.log({"step": trainer.step, **log_stats})
            print(f"[rlbench eval] epoch={i} :: {sim_stats}", flush=True)

        else:
            raise RuntimeError(
                f"eval_backend={eval_backend} not supported in this script without extra wiring."
            )
