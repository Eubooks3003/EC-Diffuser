"""
Diagnostic: can the trained diffusion model reproduce a TRAINING sample?

Loads a ckpt, takes one episode+window from the dataset, runs the full
conditional_sample loop (same as at rollout inference) conditioned on the
GT obs at t=0, action at t=0, and language, and compares the model's
predicted trajectory to the ground-truth trajectory.

Outputs:
  - Per-component L1 diff (actions / bg / particles) at each horizon step
  - Side-by-side images: GT tokens decoded vs predicted tokens decoded,
    for each horizon step, saved as a PNG

If GT and predicted are close (L1 < ~0.05 on normalized tokens, visuals
similar) -> model IS learning obs, rollout weirdness is distribution shift
If GT and predicted are very different on TRAINING data -> fundamental
training issue; bg fix may be disrupting the architecture, or particle
prediction is unlearnable for some reason.

Usage:
    cd /home/ellina/Desktop/EC-Diffuser
    export PYTHONPATH=/home/ellina/Desktop/lpwm-copy:$(pwd):$(pwd)/diffuser
    python diffuser/scripts/test_model_on_training_batch.py \
        --ckpt /home/ellina/Desktop/EC-Diffuser/data/close_jar/diffusion/rlbench_close_jar_multiview_fo/16C_dlp_adalnpint_absolute_H6_T5_seed42/ckpt/state_<N>_step<N>.pt
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import imageio.v2 as imageio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to state_*_step*.pt")
    ap.add_argument("--config", default="config.rlbench_close_jar_dlp")
    ap.add_argument("--dataset", default="close_jar")
    ap.add_argument("--num-entity", type=int, default=16)
    ap.add_argument("--input-type", default="dlp")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prefix", default="diffusion/rlbench_close_jar_multiview_fo/16C_dlp_adalnpint_absolute")
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--n-diffusion-steps", type=int, default=5)
    ap.add_argument("--logbase", default="/home/ellina/Desktop/EC-Diffuser/data")
    ap.add_argument("--lpwm-path", default="/home/ellina/Desktop/lpwm-copy")
    ap.add_argument("--episode", type=int, default=0, help="Pick episode idx to test on")
    ap.add_argument("--start-frame", type=int, default=0, help="Window start within episode")
    ap.add_argument("--out-dir", default="/home/ellina/Desktop/train_batch_test")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sys.path.insert(0, args.lpwm_path)

    # -- Hack: reuse train.py's ArgsParser to load the full config --
    import sys as _sys
    saved_argv = _sys.argv
    _sys.argv = [
        saved_argv[0],
        "--config", args.config,
        "--dataset", args.dataset,
        "--num_entity", str(args.num_entity),
        "--input_type", args.input_type,
        "--seed", str(args.seed),
        "--prefix", args.prefix,
        "--horizon", str(args.horizon),
        "--n_diffusion_steps", str(args.n_diffusion_steps),
        "--logbase", args.logbase,
    ]
    from diffuser.utils.args import ArgsParser
    parsed = ArgsParser().parse_args("diffusion")
    _sys.argv = saved_argv

    import diffuser.utils as utils
    from diffuser.utils.arrays import set_global_device
    set_global_device(args.device)

    # -- Dataset --
    dataset_config = utils.Config(
        parsed.loader,
        savepath=(parsed.savepath, "dataset_config.pkl"),
        env="",
        dataset_path=parsed.override_dataset_path,
        horizon=parsed.horizon,
        normalizer=parsed.normalizer,
        particle_normalizer=parsed.particle_normalizer,
        preprocess_fns=parsed.preprocess_fns,
        use_padding=parsed.use_padding,
        max_path_length=parsed.max_path_length,
        dataset_name=parsed.dataset,
        obs_only=parsed.obs_only,
        action_only=parsed.action_only,
        action_z_scale=getattr(parsed, "action_z_scale", 1.0),
        use_gripper_obs=getattr(parsed, "use_gripper_obs", False),
        use_bg_obs=getattr(parsed, "use_bg_obs", False),
        overfit=getattr(parsed, "overfit", False),
        max_demos=getattr(parsed, "max_demos", None),
        gripper_state_mask_ratio=getattr(parsed, "gripper_state_mask_ratio", 0.0),
        single_view=(
            parsed.input_type == "dlp" and not parsed.multiview
            and getattr(parsed, "use_views", None) is None
        ),
        clip_model_name=getattr(parsed, "clip_model_name", "openai/clip-vit-base-patch32"),
        lang_pooled=getattr(parsed, "lang_pooled", False),
        max_lang_tokens=getattr(parsed, "max_lang_tokens", 32),
        lang_device=getattr(parsed, "lang_device", "cpu"),
        use_views=getattr(parsed, "use_views", None),
        num_source_views=getattr(parsed, "num_source_views", None),
        action_normalizer=getattr(parsed, "action_normalizer", None),
    )
    dataset = dataset_config()
    obs_dim = dataset.observation_dim
    a_dim = dataset.action_dim
    gripper_dim = getattr(dataset, "gripper_dim", 0)
    bg_dim = getattr(dataset, "bg_dim", 0)
    print(f"[diag] dataset: obs_dim={obs_dim} a_dim={a_dim} gripper_dim={gripper_dim} bg_dim={bg_dim}")
    print(f"[diag] dataset fields keys: {list(dataset.fields._dict.keys())}")

    # -- Model / Diffusion --
    model_config = utils.Config(
        parsed.model,
        savepath=(parsed.savepath, "model_config.pkl"),
        features_dim=parsed.features_dim, action_dim=a_dim,
        hidden_dim=parsed.hidden_dim, projection_dim=parsed.projection_dim,
        n_head=parsed.n_heads, n_layer=parsed.n_layers, dropout=parsed.dropout,
        block_size=parsed.horizon, positional_bias=parsed.positional_bias,
        max_particles=parsed.max_particles, multiview=parsed.multiview,
        device=parsed.device, gripper_dim=gripper_dim, bg_dim=bg_dim,
        lang_dim=getattr(parsed, "lang_dim", 0),
        act_pos_dim=getattr(parsed, "act_pos_dim", 3),
        act_rot_dim=getattr(parsed, "act_rot_dim", 3),
        act_grip_dim=getattr(parsed, "act_grip_dim", 1),
        prop_pos_dim=getattr(parsed, "prop_pos_dim", 3),
        prop_rot_dim=getattr(parsed, "prop_rot_dim", 6),
        prop_grip_dim=getattr(parsed, "prop_grip_dim", 1),
    )
    diffusion_config = utils.Config(
        parsed.diffusion,
        savepath=(parsed.savepath, "diffusion_config.pkl"),
        horizon=parsed.horizon, observation_dim=obs_dim, action_dim=a_dim,
        gripper_dim=gripper_dim, bg_dim=bg_dim,
        n_timesteps=parsed.n_diffusion_steps, loss_type=parsed.loss_type,
        clip_denoised=parsed.clip_denoised, predict_epsilon=parsed.predict_epsilon,
        action_weight=parsed.action_weight, loss_weights=parsed.loss_weights,
        loss_discount=parsed.loss_discount, device=parsed.device,
    )
    model = model_config()
    diffusion = diffusion_config(model)
    diffusion.to(args.device).eval()

    sd = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    state = sd.get("ema", sd.get("model", sd))
    diffusion.load_state_dict(state, strict=False)
    print(f"[diag] loaded ckpt: {args.ckpt}")

    # -- Pick a training sample --
    # Find the index in dataset.indices that matches (episode, start_frame)
    target_idx = None
    for i, (ep_idx, s, e) in enumerate(dataset.indices):
        if ep_idx == args.episode and s == args.start_frame:
            target_idx = i; break
    if target_idx is None:
        print(f"[diag] couldn't find (ep={args.episode}, start={args.start_frame}); "
              f"using first available (ep={dataset.indices[0][0]}, start={dataset.indices[0][1]})")
        target_idx = 0

    batch = dataset[target_idx]
    print(f"[diag] sample idx={target_idx} ep={dataset.indices[target_idx][0]} start={dataset.indices[target_idx][1]}")

    # -- Prepare inputs (match how training calls model.loss) --
    def _to_tensor(x):
        if torch.is_tensor(x):
            return x.float().to(args.device)
        return torch.from_numpy(np.asarray(x)).float().to(args.device)

    gt_traj = _to_tensor(batch.trajectories).unsqueeze(0)  # (1, H, D)
    cond = {k: _to_tensor(v).unsqueeze(0) for k, v in batch.conditions.items()}
    action_cond = {k: _to_tensor(v).unsqueeze(0) for k, v in batch.action_conditions.items()}
    lang = _to_tensor(batch.lang).unsqueeze(0)
    lang_mask = _to_tensor(batch.lang_mask).unsqueeze(0)

    H = gt_traj.shape[1]
    print(f"[diag] H={H}  traj dim={gt_traj.shape[-1]}")

    # -- Run the full denoising sample loop (how rollout uses the model) --
    with torch.no_grad():
        sample = diffusion.conditional_sample(cond, lang=lang, lang_mask=lang_mask,
                                              action_cond=action_cond)
    pred_traj = sample.trajectories  # (1, H, D)
    assert pred_traj.shape == gt_traj.shape

    # -- Per-component L1 diff at each horizon step --
    obs_start = a_dim + gripper_dim + bg_dim
    bg_start = a_dim + gripper_dim
    diff = (pred_traj - gt_traj).abs().squeeze(0).cpu().numpy()  # (H, D)
    print("\n[diag] per-horizon-step L1 diff (normalized space, lower is better):")
    print(f"{'t':>3s}  {'action':>10s}  {'bg':>10s}  {'particles':>10s}")
    for t in range(H):
        a = diff[t, :a_dim].mean()
        b = diff[t, bg_start:obs_start].mean() if bg_dim > 0 else float('nan')
        p = diff[t, obs_start:].mean()
        print(f"{t:>3d}  {a:>10.5f}  {b:>10.5f}  {p:>10.5f}")

    # -- Visualize GT vs predicted particles via DLP decoder --
    from models import DLP
    cfg_path = os.path.join(os.path.dirname(parsed.override_dataset_path), "dlp_config.json")
    ckpt_path = os.path.join(os.path.dirname(parsed.override_dataset_path), "dlp_ckpt.pt")
    if not (os.path.isfile(cfg_path) and os.path.isfile(ckpt_path)):
        print(f"[diag] DLP cfg/ckpt not found next to pkl; skipping visualization")
        return
    import sys as _sys2; _sys2.path.insert(0, "/home/ellina/Desktop/EC-Diffuser/diffuser/scripts")
    from render_pkl_tokens import build_dlp_2d, decode_one_view
    with open(cfg_path) as f:
        dlp_cfg = json.load(f)
    dlp = build_dlp_2d(dlp_cfg, torch.device(args.device), DLP)
    dlp.load_state_dict(torch.load(ckpt_path, map_location=args.device, weights_only=False))

    V = 2  # front + overhead (per multiview config)
    K_per_view = int(dlp_cfg["n_kp_enc"])
    bg_per_view = bg_dim // V if bg_dim > 0 else 0
    features_dim = int(dlp_cfg.get("learned_feature_dim", 4)) + 2 + 2 + 1 + 1  # total per-particle dim

    gt_np = gt_traj.squeeze(0).cpu().numpy()     # (H, D)
    pred_np = pred_traj.squeeze(0).cpu().numpy()  # (H, D)

    # Unnormalize particles + bg (normalizer is the dataset's)
    def _decode_front(normed_traj_row):
        obs_norm = normed_traj_row[obs_start:].reshape(40, features_dim)
        obs_un = dataset.normalizer.unnormalize(obs_norm, "observations")  # (40, D)
        front = obs_un[:K_per_view]
        if bg_dim > 0:
            bg_norm = normed_traj_row[bg_start:obs_start].reshape(1, -1)
            bg_un = dataset.normalizer.unnormalize(bg_norm, "bg_features")[0]
            front_bg = bg_un[:bg_per_view]
        else:
            front_bg = np.zeros((1,), dtype=np.float32)
        return decode_one_view(dlp, front, front_bg, torch.device(args.device))

    for t in range(H):
        gt_img = _decode_front(gt_np[t])
        pr_img = _decode_front(pred_np[t])
        d_img = np.abs(gt_img.astype(int) - pr_img.astype(int)).astype(np.uint8)
        sbs = np.concatenate([gt_img, pr_img, np.clip(d_img * 3, 0, 255).astype(np.uint8)], axis=1)
        out_path = os.path.join(args.out_dir, f"t{t:02d}_gt_vs_pred.png")
        imageio.imwrite(out_path, sbs)
        print(f"  wrote {out_path}  (left=GT, middle=PRED, right=|diff|×3)")


if __name__ == "__main__":
    main()
