"""
Debug: feed the trained multitask checkpoint a *training-pkl* conditioning
(no live env, no live DLP) for close_jar's above-jar -> descent keypose pair,
and compare the predicted next-keypose action to the dataset ground truth.

If predicted z lands near 0.758 (training target) -> live-env eval is failing
because live DLP encoding != pkl-stored DLP encoding (cond distribution shift).
If predicted z lands near 0.724 (live eval value) -> model genuinely
hasn't converged for this keypose pair.

Run from /home/ellina/Desktop/EC-Diffuser-2D with the same PYTHONPATH the eval
worker uses.
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch

# Mirror eval_paper_rlbench_2d.py's path setup
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DIFFUSER_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
EC_DIFFUSER_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
for p in (DIFFUSER_ROOT, EC_DIFFUSER_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import diffuser.utils as utils
from diffuser.utils.args import ArgsParser


def main():
    # Build args from the multitask multiview config exactly as the eval worker does.
    sys.argv = [
        sys.argv[0],
        "--config", "config.rlbench_multitask_keypose_multiview_dlp",
        "--dataset", "multitask",
        "--num_entity", "16",
        "--input_type", "dlp",
        "--seed", "42",
    ]
    args = ArgsParser().parse_args("diffusion")
    print(f"[dbg] savepath={args.savepath}", flush=True)
    print(f"[dbg] type(override_dataset_path) = {type(args.override_dataset_path).__name__}", flush=True)
    print(f"[dbg] override_dataset_path = {args.override_dataset_path}", flush=True)
    print(f"[dbg] dataset_path = {getattr(args, 'dataset_path', '<none>')}", flush=True)

    # ---------- dataset ----------
    print("[dbg] building dataset (this loads + normalizes all 10 task pkls)...", flush=True)
    dataset_config = utils.Config(
        args.loader,
        savepath=(args.savepath, "dataset_config.pkl"),
        env="",
        horizon=args.horizon,
        normalizer=args.normalizer,
        particle_normalizer=args.particle_normalizer,
        preprocess_fns=args.preprocess_fns,
        use_padding=args.use_padding,
        max_path_length=args.max_path_length,
        dataset_path=args.override_dataset_path,
        dataset_name=args.dataset,
        obs_only=args.obs_only,
        action_only=args.action_only,
        keypose_mode=getattr(args, "keypose_mode", False),
        use_gripper_obs=getattr(args, "use_gripper_obs", False),
        use_bg_obs=getattr(args, "use_bg_obs", False),
        action_z_scale=getattr(args, "action_z_scale", 1.0),
        overfit=getattr(args, "overfit", False),
        max_demos=getattr(args, "max_demos", None),
        gripper_state_mask_ratio=getattr(args, "gripper_state_mask_ratio", 0.0),
        single_view=(args.input_type == "dlp" and not args.multiview
                     and getattr(args, "use_views", None) is None),
        clip_model_name=getattr(args, "clip_model_name", "openai/clip-vit-base-patch32"),
        lang_pooled=getattr(args, "lang_pooled", False),
        max_lang_tokens=getattr(args, "max_lang_tokens", 32),
        lang_device=getattr(args, "lang_device", "cpu"),
        use_views=getattr(args, "use_views", None),
        num_source_views=getattr(args, "num_source_views", None),
        action_normalizer=getattr(args, "action_normalizer", None),
    )
    dataset = dataset_config()

    # ---------- model + diffusion ----------
    obs_dim = dataset.observation_dim
    act_dim = dataset.action_dim
    grip_dim = getattr(dataset, "gripper_dim", 0)
    bg_dim = getattr(dataset, "bg_dim", 0)

    # Mirror eval_paper_rlbench_2d.py:256-282 exactly — singular n_head/n_layer,
    # block_size=horizon, and use_cond_tokens=keypose_mode are all required for
    # the trained checkpoint's parameter shapes to match.
    model_config = utils.Config(
        args.model,
        savepath=(args.savepath, "model_config.pkl"),
        features_dim=args.features_dim,
        action_dim=act_dim,
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
        gripper_dim=grip_dim,
        bg_dim=bg_dim,
        lang_dim=getattr(args, "lang_dim", 0),
        act_pos_dim=getattr(args, "act_pos_dim", 3),
        act_rot_dim=getattr(args, "act_rot_dim", 3),
        act_grip_dim=getattr(args, "act_grip_dim", 1),
        prop_pos_dim=getattr(args, "prop_pos_dim", 3),
        prop_rot_dim=getattr(args, "prop_rot_dim", 6),
        prop_grip_dim=getattr(args, "prop_grip_dim", 1),
        split_action_tokens=getattr(args, "split_action_tokens", None),
        use_cond_tokens=getattr(args, "keypose_mode", False),
    )
    model = model_config()

    diffusion_config = utils.Config(
        args.diffusion,
        savepath=(args.savepath, "diffusion_config.pkl"),
        horizon=args.horizon,
        observation_dim=obs_dim,
        action_dim=act_dim,
        gripper_dim=grip_dim,
        bg_dim=bg_dim,
        n_timesteps=args.n_diffusion_steps,
        loss_type=args.loss_type,
        clip_denoised=args.clip_denoised,
        predict_epsilon=args.predict_epsilon,
        action_weight=args.action_weight,
        loss_weights=args.loss_weights,
        loss_discount=args.loss_discount,
        device=args.device,
        keypose_mode=getattr(args, "keypose_mode", False),
    )
    diffusion = diffusion_config(model)

    # Locate latest ckpt
    import glob
    ckpts = sorted(glob.glob(os.path.join(args.savepath, "ckpt", "state_*.pt")))
    assert ckpts, f"no ckpts in {args.savepath}/ckpt"
    ckpt_path = ckpts[-1]
    print(f"[dbg] ckpt: {ckpt_path}", flush=True)

    sd = torch.load(ckpt_path, map_location=args.device)
    state = sd.get("ema", sd.get("model", sd))

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
    trainer.ema_model.load_state_dict(state, strict=True)
    trainer.ema_model.eval()
    device = args.device

    # ---------- pick the close_jar ep=0 above-jar -> descent transition ----------
    # close_jar is TASK_PKLS[0], so episodes [0..99] are close_jar.
    # After the frame-0 prepend (sequence.py:170-176), keypose_indices[ep, 0]=0
    # and the original keyposes shift to [1, 2, 3, ...].
    path_ind = 0
    expected_cond_frame = 48   # raw t for above-jar in ep=0
    expected_target_frame = 64 # raw t for descent  in ep=0
    kp_start = 2  # target = 2nd keypose after prepend (i.e., descent)
    cond_pos = kp_start - 1

    cond_frame   = int(dataset.keypose_indices[path_ind, cond_pos])
    target_frame = int(dataset.keypose_indices[path_ind, kp_start])
    print(f"[dbg] path_ind={path_ind}  cond_frame={cond_frame} (expect {expected_cond_frame})  "
          f"target_frame={target_frame} (expect {expected_target_frame})", flush=True)
    print(f"[dbg] dataset.keypose_indices[0,:8] = {dataset.keypose_indices[0,:8].tolist()}")
    print(f"[dbg] dataset.n_keyposes[0] = {int(dataset.n_keyposes[0])}")
    _lang_first = dataset.fields._dict.get("language", [None])[path_ind]
    print(f"[dbg] dataset.language[0] = {_lang_first!r}")
    print(f"[dbg] number of demos in dataset = {dataset.fields._count}")

    # Pull dataset-canonical normed cond + target.
    cond_obs = dataset.fields.normed_observations[path_ind, cond_frame]      # (K, D) post-view-slice
    cond_grip = dataset.fields.normed_gripper_state[path_ind, cond_frame]    # (10,)
    cond_bg  = dataset.fields.normed_bg_features[path_ind, cond_frame]       # (8,)
    target_action_normed = dataset.fields.normed_actions[path_ind, target_frame]  # (10,)
    target_action_raw    = dataset.normalizer.unnormalize(
        target_action_normed[None], "actions")[0]

    # cond[0] order: [gripper, bg, obs.flatten()]  -- mirrors training.py:933-939
    cond_vec = np.concatenate([cond_grip, cond_bg, cond_obs.reshape(-1)], axis=-1)
    cond_t = torch.from_numpy(cond_vec).float().to(device).unsqueeze(0)  # (1, gripper+bg+K*D)
    cond = {0: cond_t}

    # ---------- language ----------
    # Pull the language string for ep=0 from the merged dataset.
    lang_strs = dataset.fields._dict.get("language", None) or []
    lang_str = lang_strs[path_ind] if path_ind < len(lang_strs) else "close the jar"
    if isinstance(lang_str, (list, tuple)):
        lang_str = lang_str[0]
    print(f"[dbg] language for ep=0: {lang_str!r}", flush=True)

    from diffuser.sampling.policies import LanguageConditionedPolicy
    policy = LanguageConditionedPolicy(
        diffusion_model=trainer.ema_model,
        normalizer=dataset.normalizer,
        preprocess_fns=[],
        clip_model_name=getattr(args, "clip_model_name", "openai/clip-vit-base-patch32"),
        lang_pooled=getattr(args, "lang_pooled", False),
        max_lang_tokens=getattr(args, "max_lang_tokens", 32),
    )
    policy.set_instruction(lang_str)
    lang = policy._lang_tokens.to(device)[:1]
    lang_mask = policy._lang_mask.to(device)[:1]

    # ---------- run ----------
    keypose_mode = bool(getattr(args, "keypose_mode", False))
    action_cond = {} if keypose_mode else None
    with torch.no_grad():
        sample = trainer.ema_model(cond, lang=lang, lang_mask=lang_mask,
                                   action_cond=action_cond, verbose=False)
    traj = sample.trajectories[0]                # (H, a_dim+grip+bg+obs)
    a_norm_pred = traj[0, :act_dim].cpu().numpy()
    a_pred = dataset.normalizer.unnormalize(a_norm_pred[None], "actions")[0]

    print()
    print(f"  TARGET (training)  pos=({target_action_raw[0]:+.3f}, "
          f"{target_action_raw[1]:+.3f}, {target_action_raw[2]:+.3f})  "
          f"grip={target_action_raw[9]:+.3f}")
    print(f"  PREDICTED (model)  pos=({a_pred[0]:+.3f}, {a_pred[1]:+.3f}, "
          f"{a_pred[2]:+.3f})  grip={a_pred[9]:+.3f}")
    print(f"  delta z = {a_pred[2] - target_action_raw[2]:+.4f} m")
    print(f"  delta grip = {a_pred[9] - target_action_raw[9]:+.4f}")

    # Live-eval predicted z for this same scene was 0.724.
    print()
    print(f"  Live-eval predicted z was +0.724 (cond from live DLP encode)")
    print(f"  This  test  predicted z is  {a_pred[2]:+.3f} (cond from pkl-stored encode)")
    if abs(a_pred[2] - 0.758) < 0.015:
        print("  -> model converged on training cond. Bug is live DLP != pkl DLP.")
    elif abs(a_pred[2] - 0.724) < 0.015:
        print("  -> model also wrong on training cond. Bug is undertraining.")
    else:
        print("  -> something else: predicted z is in neither bucket")


if __name__ == "__main__":
    main()
