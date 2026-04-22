"""
Decompose the diffusion model's prediction error on a training window
by horizon-step AND by particle feature slice.

Purpose: rule in/out whether "particles L1 ~0.33" is uniform across all
token dims or concentrated in a subset (z xy / z_scale / z_depth / obj_on
/ z_feat). Also verifies apply_conditioning really pins a0 and obs@t=0.

Usage (conda ecdiffuser):
    cd /home/ellina/Desktop/EC-Diffuser
    export PYTHONPATH=/home/ellina/Desktop/lpwm-copy:$(pwd):$(pwd)/diffuser
    python diffuser/scripts/diag_loss_decomp.py --ckpt <path>
"""
import argparse
import os
import sys

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
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
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--n-windows", type=int, default=4,
                    help="Average over this many training windows for stability")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    sys.path.insert(0, args.lpwm_path)

    # Reuse train.py's ArgsParser
    saved_argv = sys.argv
    sys.argv = [
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
    sys.argv = saved_argv

    import diffuser.utils as utils
    from diffuser.utils.arrays import set_global_device
    set_global_device(args.device)

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
        single_view=(parsed.input_type == "dlp" and not parsed.multiview
                     and getattr(parsed, "use_views", None) is None),
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
    features_dim = int(parsed.features_dim)

    # Infer K, n_views
    K = obs_dim // features_dim                      # total particles across views
    n_views = 2 if getattr(parsed, "multiview", False) else 1
    K_per_view = K // n_views
    bg_per_view = bg_dim // n_views if bg_dim > 0 else 0

    print(f"[diag] obs_dim={obs_dim} a_dim={a_dim} gripper_dim={gripper_dim} "
          f"bg_dim={bg_dim} K={K} K_per_view={K_per_view} features_dim={features_dim}")

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

    # Pick N training windows (consecutive episodes, start=0)
    picks = []
    for ep in range(args.n_windows):
        for i, (e, s, _) in enumerate(dataset.indices):
            if e == ep and s == args.start_frame:
                picks.append(i); break
    if not picks:
        picks = [0]
    print(f"[diag] picked {len(picks)} windows: {picks}")

    def _t(x):
        return torch.from_numpy(np.asarray(x)).float().to(args.device) if not torch.is_tensor(x) \
               else x.float().to(args.device)

    # ------------------------------------------------------------------
    # (A) Pinning verification: run ONE forward of p_losses's first steps
    #     manually and inspect x_noisy BEFORE vs AFTER apply_conditioning.
    # ------------------------------------------------------------------
    from diffuser.models.helpers import apply_conditioning

    batch = dataset[picks[0]]
    x_start = _t(batch.trajectories).unsqueeze(0)
    cond = {k: _t(v).unsqueeze(0) for k, v in batch.conditions.items()}
    action_cond = {k: _t(v).unsqueeze(0) for k, v in batch.action_conditions.items()}

    # Simulate q_sample at max noise (t = n_timesteps - 1)
    t_max = torch.full((1,), parsed.n_diffusion_steps - 1, device=args.device, dtype=torch.long)
    noise = torch.randn_like(x_start)
    x_noisy = diffusion.q_sample(x_start=x_start, t=t_max, noise=noise)

    pre_a = x_noisy[0, 0, :a_dim].detach().cpu().numpy()
    pre_obs = x_noisy[0, 0, a_dim:a_dim + 8].detach().cpu().numpy()  # first 8 obs dims

    x_noisy_pinned = apply_conditioning(x_noisy.clone(), cond, a_dim, action_conditions=action_cond)

    post_a = x_noisy_pinned[0, 0, :a_dim].detach().cpu().numpy()
    post_obs = x_noisy_pinned[0, 0, a_dim:a_dim + 8].detach().cpu().numpy()
    tgt_a = x_start[0, 0, :a_dim].detach().cpu().numpy()
    tgt_obs = x_start[0, 0, a_dim:a_dim + 8].detach().cpu().numpy()
    ac = action_cond[0][0].detach().cpu().numpy()
    c0 = cond[0][0][:8].detach().cpu().numpy()

    print("\n" + "=" * 70)
    print("(A) apply_conditioning verification")
    print("=" * 70)
    print(f"action_cond[0][0,:a_dim]         = {ac.round(4)}")
    print(f"cond[0][0, :8]                   = {c0.round(4)}")
    print(f"x_noisy[0,0,:a_dim]    BEFORE    = {pre_a.round(4)}  (should be noised)")
    print(f"x_noisy[0,0,:a_dim]    AFTER     = {post_a.round(4)}  (should == action_cond)")
    print(f"x_start[0,0,:a_dim]              = {tgt_a.round(4)}  (should match AFTER if rebind identity)")
    print(f"|AFTER - action_cond|            = {np.abs(post_a - ac).max():.2e}  (0 == pinned)")
    print(f"|AFTER - x_start|                = {np.abs(post_a - tgt_a).max():.2e}  (0 == rebind works)")
    print()
    print(f"x_noisy[0,0,a_dim:+8]  BEFORE    = {pre_obs.round(4)}  (should be noised)")
    print(f"x_noisy[0,0,a_dim:+8]  AFTER     = {post_obs.round(4)}  (should == cond[0][:8])")
    print(f"x_start[0,0,a_dim:+8]            = {tgt_obs.round(4)}")
    print(f"|AFTER - cond|                   = {np.abs(post_obs - c0).max():.2e}  (0 == obs@t=0 pinned)")
    print(f"|AFTER - x_start|                = {np.abs(post_obs - tgt_obs).max():.2e}")

    # ------------------------------------------------------------------
    # (B) Full-sample per-horizon-step, per-feature-slice decomposition
    # ------------------------------------------------------------------
    obs_start = a_dim + gripper_dim + bg_dim
    bg_start = a_dim + gripper_dim

    # feature slices inside each particle token (10D):
    #   z(0:2), z_scale(2:4), z_depth(4:5), obj_on(5:6), z_feat(6:10)
    feat_slices = {
        "z(xy)":   (0, 2),
        "z_scale": (2, 4),
        "z_depth": (4, 5),
        "obj_on":  (5, 6),
        "z_feat":  (6, 10),
    }

    per_h_actions = []
    per_h_bg = []
    per_h_part_total = []
    per_h_part_by_feat = {k: [] for k in feat_slices}

    with torch.no_grad():
        for p_idx in picks:
            b = dataset[p_idx]
            gt = _t(b.trajectories).unsqueeze(0)
            c = {k: _t(v).unsqueeze(0) for k, v in b.conditions.items()}
            ac_ = {k: _t(v).unsqueeze(0) for k, v in b.action_conditions.items()}
            lang = _t(b.lang).unsqueeze(0) if b.lang is not None else None
            lm = _t(b.lang_mask).unsqueeze(0) if b.lang_mask is not None else None
            sample = diffusion.conditional_sample(c, lang=lang, lang_mask=lm, action_cond=ac_)
            pr = sample.trajectories
            diff = (pr - gt).abs().squeeze(0).cpu().numpy()   # (H, D_total)
            per_h_actions.append(diff[:, :a_dim].mean(-1))    # (H,)
            if bg_dim > 0:
                per_h_bg.append(diff[:, bg_start:obs_start].mean(-1))  # (H,)
            parts_diff = diff[:, obs_start:].reshape(diff.shape[0], K, features_dim)  # (H, K, D)
            per_h_part_total.append(parts_diff.mean(axis=(1, 2)))  # (H,)
            for name, (a, b_) in feat_slices.items():
                per_h_part_by_feat[name].append(parts_diff[:, :, a:b_].mean(axis=(1, 2)))

    per_h_actions = np.mean(per_h_actions, axis=0)
    per_h_bg = np.mean(per_h_bg, axis=0) if per_h_bg else None
    per_h_part_total = np.mean(per_h_part_total, axis=0)
    for k in feat_slices:
        per_h_part_by_feat[k] = np.mean(per_h_part_by_feat[k], axis=0)

    print("\n" + "=" * 70)
    print(f"(B) Per-horizon L1 averaged over {len(picks)} training window(s)")
    print("=" * 70)
    hdr = f"{'t':>3}  {'action':>8}  {'bg':>8}  {'part':>8}"
    for name in feat_slices:
        hdr += f"  {name:>8}"
    print(hdr)
    H = len(per_h_actions)
    for t in range(H):
        row = f"{t:>3}  {per_h_actions[t]:>8.5f}"
        row += f"  {per_h_bg[t]:>8.5f}" if per_h_bg is not None else f"  {'-':>8}"
        row += f"  {per_h_part_total[t]:>8.5f}"
        for name in feat_slices:
            row += f"  {per_h_part_by_feat[name][t]:>8.5f}"
        print(row)

    # Quick sanity ceilings: what would "predict current = next" give?
    print("\n" + "=" * 70)
    print("(C) Data baseline: L1 between obs[t] and obs[t+1] on normed tokens")
    print("    (lower bound for what the model should be able to beat)")
    print("=" * 70)
    normed_obs = dataset.fields.normed_observations  # (N, T, K*D)
    path_lens = dataset.fields.path_lengths
    deltas_by_feat = {k: [] for k in feat_slices}
    for ep in range(min(20, normed_obs.shape[0])):
        L = int(path_lens[ep])
        if L < 2:
            continue
        ob = normed_obs[ep, :L].reshape(L, K, features_dim)
        d = np.abs(ob[1:] - ob[:-1])   # (L-1, K, D)
        for name, (a, b_) in feat_slices.items():
            deltas_by_feat[name].append(d[:, :, a:b_].mean())
    for name in feat_slices:
        vals = deltas_by_feat[name]
        if vals:
            print(f"  {name:>8}: mean |obs[t+1]-obs[t]| = {np.mean(vals):.5f}")

    # Per-slot stability: is slot i's z smooth over time?
    print("\n" + "=" * 70)
    print("(D) Slot-identity probe: per-slot frame-to-frame z-displacement vs")
    print("    instantaneous slot spread (across slots at one frame)")
    print("    delta << spread -> slots stable  |  delta ~= spread -> slots shuffle")
    print("=" * 70)
    per_slot_z_delta = []
    per_frame_z_spread = []
    for ep in range(min(20, normed_obs.shape[0])):
        L = int(path_lens[ep])
        if L < 2:
            continue
        ob = normed_obs[ep, :L].reshape(L, K, features_dim)
        z = ob[:, :, 0:2]  # (L, K, 2)
        per_slot_z_delta.append(np.linalg.norm(z[1:] - z[:-1], axis=-1).mean())
        per_frame_z_spread.append(z.std(axis=1).mean())
    print(f"  mean per-slot frame-to-frame z-delta    = {np.mean(per_slot_z_delta):.5f}")
    print(f"  mean instantaneous slot-spread (std)    = {np.mean(per_frame_z_spread):.5f}")
    ratio = np.mean(per_slot_z_delta) / (np.mean(per_frame_z_spread) + 1e-9)
    print(f"  ratio delta/spread                      = {ratio:.3f}")
    print(f"  interpretation: {'slots SHUFFLE (chamfer helps)' if ratio > 0.5 else 'slots STABLE (L1 should work)'}")


if __name__ == "__main__":
    main()
