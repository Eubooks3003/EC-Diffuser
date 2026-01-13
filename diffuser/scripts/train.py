import warnings
warnings.filterwarnings("ignore")

import os
import sys
import wandb
import torch

import diffuser.utils as utils
from diffuser.utils.arrays import set_global_device
from diffuser.utils.args import ArgsParser


# -----------------------------------------------------------------------------#
#                        make lpwm-dev importable                               #
# -----------------------------------------------------------------------------#

# train.py is usually at:  EC-Diffuser/diffuser/scripts/train.py
# lpwm-dev is sibling of EC-Diffuser:  .../Code/lpwm-dev
LPWM_DEV = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "lpwm-dev")
)
if os.path.isdir(LPWM_DEV) and LPWM_DEV not in sys.path:
    sys.path.append(LPWM_DEV)


# -----------------------------------------------------------------------------#
#                          LPWM DLP loading (cfg + ckpt)                        #
# -----------------------------------------------------------------------------#

def build_dlp_from_cfg(cfg, device, DLPClass):
    # mirrors your preprocessing constructor exactly
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


def load_dlp_lpwm(dlp_cfg_path: str, dlp_ckpt_path: str, device: str):
    # from lpwm-dev
    from utils.util_func import get_config
    from utils.log_utils import load_checkpoint
    from voxel_models import DLP as DLPClass

    dev = torch.device(device)
    cfg = get_config(dlp_cfg_path)
    model = build_dlp_from_cfg(cfg, dev, DLPClass)
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
)

render_config = utils.Config(
    args.renderer,
    savepath=(args.savepath, "render_config.pkl"),
    env=None,
    num_entity=args.num_entity,
    particle_dim=args.features_dim,
    single_view=(args.input_type == "dlp" and not args.multiview),
)

dataset = dataset_config()
renderer = render_config()

observation_dim = dataset.observation_dim
action_dim = dataset.action_dim


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
)

diffusion_config = utils.Config(
    args.diffusion,
    savepath=(args.savepath, "diffusion_config.pkl"),
    horizon=args.horizon,
    observation_dim=observation_dim,
    action_dim=action_dim,
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
wandb_run.name = args.savepath.split("/")[-1]


# -----------------------------------------------------------------------------#
#                               mimicgen eval wiring                            #
# -----------------------------------------------------------------------------#

dlp_model = None
dlp_cfg = None
make_env_fn = None
calib_h5_path = None

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
    dlp_model, dlp_cfg = load_dlp_lpwm(dlp_cfg_path, dlp_ckpt, args.device)

    def make_env_fn():
        from diffuser.eval_utils import setup_mimicgen_env
        return setup_mimicgen_env(args)


# -----------------------------------------------------------------------------#
#                                  main loop                                   #
# -----------------------------------------------------------------------------#

n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

for i in range(n_epochs):
    print(f"Epoch {i} / {n_epochs} | {args.savepath}", flush=True)
    trainer.train(n_train_steps=args.n_steps_per_epoch)

    # eval AFTER each epoch; with eval_freq=1 it runs every epoch
    if do_eval and ((i + 1) % eval_freq == 0):
        print(f"[eval] starting {eval_backend} eval at epoch={i} step={trainer.step}", flush=True)

        if eval_backend == "mimicgen":
            sim_stats = trainer.eval_mimicgen_rollouts(
                make_env_fn=make_env_fn,
                dlp_model=dlp_model,
                calib_h5_path=calib_h5_path,
                n_episodes=getattr(args, "mimicgen_eval_episodes", 5),
                max_steps=getattr(args, "mimicgen_max_steps", 50),
                bounds_xyz=getattr(args, "mimicgen_bounds_xyz", ((-2, 2), (-2, 2), (-0.2, 2.5))),
                grid_dhw=getattr(args, "mimicgen_grid_dhw", (64, 64, 64)),
                cams=getattr(args, "mimicgen_cams", ("agentview", "sideview", "robot0_eye_in_hand")),
                pixel_stride=getattr(args, "mimicgen_pixel_stride", 2),
                goal_from_env_fn=getattr(args, "goal_from_env_fn", None),
            )

            # avoid double-prefix if eval returns sim/... already
            log_stats = {k if k.startswith("sim/") else f"sim/{k}": v for k, v in sim_stats.items()}
            wandb.log({"step": trainer.step, **log_stats})
            print(f"[mimicgen eval] epoch={i} :: {sim_stats}", flush=True)

        else:
            raise RuntimeError(
                f"eval_backend={eval_backend} not supported in this script without extra wiring."
            )
