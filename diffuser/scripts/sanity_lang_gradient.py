"""
Pre-training sanity: is the language path actually wired into the action output?

Builds the denoiser from the same config as train.py, runs one forward+backward
on random inputs, and checks:
  (1) lang_projection and lang_encoding receive non-zero gradients.
  (2) Perturbing the language input measurably changes the action output.

If either check fails, the self-attn-over-language wiring has a bug and running
a full training job would be wasted compute.

Usage (same args style as eval_rlbench.py):
    python diffuser/scripts/sanity_lang_gradient.py \
        --config config.rlbench_close_jar_dlp \
        --dataset close_jar --num_entity 16 --input_type dlp --seed 42
"""
import sys

import numpy as np
import torch

import diffuser.utils as utils


def _build_args(raw_argv):
    from diffuser.utils.args import ArgsParser
    sys.argv = [sys.argv[0]] + list(raw_argv)
    return ArgsParser().parse_args("diffusion")


def main(argv):
    args = _build_args(argv)
    device = args.device
    print(f"[sanity_lang_gradient] savepath={args.savepath} device={device}", flush=True)

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
    action_dim = dataset.action_dim
    gripper_dim = getattr(dataset, "gripper_dim", 0)
    bg_dim = getattr(dataset, "bg_dim", 0)

    model_config = utils.Config(
        args.model, savepath=(args.savepath, "model_config.pkl"),
        features_dim=args.features_dim, action_dim=action_dim,
        hidden_dim=args.hidden_dim, projection_dim=args.projection_dim,
        n_head=args.n_heads, n_layer=args.n_layers, dropout=args.dropout,
        block_size=args.horizon, positional_bias=args.positional_bias,
        max_particles=args.max_particles, multiview=args.multiview,
        device=device, gripper_dim=gripper_dim, bg_dim=bg_dim,
        lang_dim=getattr(args, "lang_dim", 0),
        act_pos_dim=getattr(args, "act_pos_dim", 3),
        act_rot_dim=getattr(args, "act_rot_dim", 3),
        act_grip_dim=getattr(args, "act_grip_dim", 1),
        prop_pos_dim=getattr(args, "prop_pos_dim", 3),
        prop_rot_dim=getattr(args, "prop_rot_dim", 6),
        prop_grip_dim=getattr(args, "prop_grip_dim", 1),
    )
    model = model_config().to(device).train()

    K = int(args.max_particles)
    D = int(args.features_dim)
    lang_dim = int(getattr(args, "lang_dim", 0))
    max_lang_tokens = int(getattr(args, "max_lang_tokens", 32))
    horizon = int(args.horizon)
    transition_dim = action_dim + gripper_dim + bg_dim + K * D

    print(f"[shape] horizon={horizon} action_dim={action_dim} gripper_dim={gripper_dim} "
          f"bg_dim={bg_dim} K={K} D={D} lang_dim={lang_dim} max_lang_tokens={max_lang_tokens}")
    print(f"[shape] transition_dim={transition_dim} (per-timestep flat)")
    print(f"[shape] expected sequence N = 6 + K + L_lang = {6 + K + max_lang_tokens}")

    # ------------------------------------------------------------------ grad test
    bs = 2
    x = torch.randn(bs, horizon, transition_dim, device=device)
    t = torch.randint(0, 5, (bs,), device=device)
    lang = torch.randn(bs, max_lang_tokens, lang_dim, device=device) if lang_dim > 0 else None

    out = model(x, cond=None, time=t, lang=lang)
    assert out.shape == x.shape, f"out {out.shape} != x {x.shape}"

    loss = out[:, :, :action_dim].pow(2).mean()
    loss.backward()

    print("\n[gradient norms] (zero => parameter not in backward path)")
    print("-" * 70)
    watch = [
        "lang_projection", "lang_encoding",
        "particle_projection", "a_pos_projection",
        "particle_transformer.blocks.0.attn.query",
        "particle_transformer.blocks.0.attn.key",
        "particle_transformer.blocks.0.attn.value",
        "particle_transformer.blocks.0.mlp",
    ]
    lang_grad_ok = True
    for name, p in model.named_parameters():
        if not any(k in name for k in watch):
            continue
        g = p.grad
        if g is None:
            status = "NONE"
            is_zero = True
        else:
            is_zero = bool(g.abs().max().item() == 0)
            status = f"{g.norm().item():.4e}" if not is_zero else "ZERO"
        flag = "" if not is_zero else "  !!"
        if "lang_" in name and is_zero:
            lang_grad_ok = False
        print(f"  {name:65s}  {status}{flag}")
    print("-" * 70)

    # ------------------------------------------------------------------ perturbation test
    model.zero_grad()
    model.eval()
    with torch.no_grad():
        lang_a = lang
        lang_b = torch.randn_like(lang_a) if lang_a is not None else None
        out_a = model(x, cond=None, time=t, lang=lang_a)
        out_b = model(x, cond=None, time=t, lang=lang_b)
        diff_action = (out_a[:, :, :action_dim] - out_b[:, :, :action_dim]).abs().mean().item()
        diff_particles = (out_a[:, :, action_dim + gripper_dim + bg_dim:] -
                          out_b[:, :, action_dim + gripper_dim + bg_dim:]).abs().mean().item()
    print(f"\n[perturbation] two random lang tensors, same x and t:")
    print(f"  |action_out_a - action_out_b|    = {diff_action:.4e}")
    print(f"  |particle_out_a - particle_out_b| = {diff_particles:.4e}")
    perturbation_ok = diff_action > 1e-4

    # ------------------------------------------------------------------ verdict
    print("\n" + "=" * 70)
    if lang_grad_ok and perturbation_ok:
        print("[OK] Language is wired into the action output and receives gradient.")
        print("     At-init behavior may still look ~uniform over tokens — that is")
        print("     expected for random weights. Train briefly, then run")
        print("     inspect_attention.py to see whether the model learns to attend")
        print("     to real language tokens over padding.")
    else:
        print("[FAIL] Language wiring is broken:")
        if not lang_grad_ok:
            print("   - lang_* parameters have zero/None grad. They are not in the")
            print("     backward path of the action loss.")
        if not perturbation_ok:
            print(f"   - Changing the language input barely changes action output")
            print(f"     ({diff_action:.2e}). Action doesn't depend on language.")
        print("   Fix before launching any long training run.")
    print("=" * 70)


if __name__ == "__main__":
    main(sys.argv[1:])
