"""
Attention inspection: load a trained checkpoint, run one denoising step on a
real observation+language pair, and print how much attention the action tokens
pay to language tokens vs particle / proprio / action tokens.

Answers: "is the policy actually attending to real language tokens?"

After the self-attn-over-language patch, AdaLNPINTDenoiser token order is:
    [a_pos, a_rot, a_grip, p_pos, p_rot, p_grip, particles(K), lang(L_lang)]

Per layer we report, from the SELF-attention matrix, mean weight flowing from
the three action-query rows to each group:
    action -> action       (self-bias within the 3 action tokens)
    action -> proprio      (3 proprio tokens)
    action -> particles    (K particle tokens)
    action -> lang_real    (first n_valid lang positions with real content)
    action -> lang_pad     (remaining padded lang positions)

If lang_real mass >> uniform baseline and >> lang_pad, the model is using
language. If lang_real ~ lang_pad, the policy is not distinguishing real
content from padding.

Usage (identical args to eval_rlbench.py):
    xvfb-run -a python diffuser/scripts/inspect_attention.py \\
        --config config.rlbench_close_jar_dlp \\
        --dataset close_jar --num_entity 16 --input_type dlp --seed 42 \\
        --ckpt_step latest
"""
import argparse
import os
import sys

import numpy as np
import torch

import diffuser.utils as utils


def _build_args(raw_argv):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--ckpt_step", default="latest")
    pre.add_argument("--n_samples", type=int, default=3,
                     help="number of (obs, lang) pairs to average over")
    ours, rest = pre.parse_known_args(raw_argv)

    from diffuser.utils.args import ArgsParser
    sys.argv = [sys.argv[0]] + rest
    args = ArgsParser().parse_args("diffusion")
    args._ckpt_step = ours.ckpt_step
    args._n_samples = ours.n_samples
    return args


def main(argv):
    args = _build_args(argv)
    print(f"[inspect_attention] savepath = {args.savepath}", flush=True)

    # Build dataset (for normalizer + sample observations)
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

    # Model / diffusion
    model_config = utils.Config(
        args.model, savepath=(args.savepath, "model_config.pkl"),
        features_dim=args.features_dim, action_dim=action_dim,
        hidden_dim=args.hidden_dim, projection_dim=args.projection_dim,
        n_head=args.n_heads, n_layer=args.n_layers, dropout=args.dropout,
        block_size=args.horizon, positional_bias=args.positional_bias,
        max_particles=args.max_particles, multiview=args.multiview,
        device=args.device, gripper_dim=gripper_dim, bg_dim=bg_dim,
        lang_dim=getattr(args, "lang_dim", 0),
        act_pos_dim=getattr(args, "act_pos_dim", 3),
        act_rot_dim=getattr(args, "act_rot_dim", 3),
        act_grip_dim=getattr(args, "act_grip_dim", 1),
        prop_pos_dim=getattr(args, "prop_pos_dim", 3),
        prop_rot_dim=getattr(args, "prop_rot_dim", 6),
        prop_grip_dim=getattr(args, "prop_grip_dim", 1),
    )
    diffusion_config = utils.Config(
        args.diffusion, savepath=(args.savepath, "diffusion_config.pkl"),
        horizon=args.horizon, observation_dim=observation_dim, action_dim=action_dim,
        gripper_dim=gripper_dim, bg_dim=bg_dim,
        n_timesteps=args.n_diffusion_steps, loss_type=args.loss_type,
        clip_denoised=args.clip_denoised, predict_epsilon=args.predict_epsilon,
        action_weight=args.action_weight, loss_weights=args.loss_weights,
        loss_discount=args.loss_discount, device=args.device,
    )
    model = model_config()
    diffusion = diffusion_config(model)
    diffusion.to(args.device).eval()

    # Load ckpt
    ckpt_dir = os.path.join(args.savepath, "ckpt")
    files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    def _step_of(f):
        try: return int(f.rsplit("_step", 1)[-1].split(".")[0])
        except: return -1
    if args._ckpt_step == "latest":
        files.sort(key=_step_of)
        ckpt_path = os.path.join(ckpt_dir, files[-1])
    else:
        ckpt_path = os.path.join(ckpt_dir,
            [f for f in files if f.endswith(f"_step{args._ckpt_step}.pt")][0])
    print(f"[inspect_attention] loading {ckpt_path}", flush=True)
    sd = torch.load(ckpt_path, map_location=args.device)
    state = sd.get("ema", sd.get("model", sd))
    diffusion.load_state_dict(state, strict=False)

    # ------------------------------------------------------------------
    # Sample (obs, lang) batch from dataset, run 1 denoising step with
    # return_attention=True
    # ------------------------------------------------------------------
    from diffuser.models.lang import CLIPTextEncoder
    clip_enc = CLIPTextEncoder(
        model_name=getattr(args, "clip_model_name", "openai/clip-vit-base-patch32"),
        device="cpu",
        return_pooled=bool(getattr(args, "lang_pooled", False)),
    )
    max_lang_tokens = int(getattr(args, "max_lang_tokens", 32))

    # Sanity check: are CLIP's pad embeddings actually distinct per position?
    # If they ARE, the language fix is meaningful. If they're IDENTICAL, the
    # fix has no effect (and we'd need an explicit positional embedding on top).
    _tokens, _mask = clip_enc.encode(["close the red jar"])
    _raw = _tokens[0, :max_lang_tokens].float()
    _valid_raw = int(_mask[0].sum().item())
    _real = _raw[:_valid_raw]
    _pad = _raw[_valid_raw:]
    print(f"[sanity] raw CLIP output (before any MLP/projection):")
    print(f"  real tokens [:{_valid_raw}] norm: mean={_real.norm(dim=-1).mean():.4f}")
    print(f"  pad tokens  [{_valid_raw}:]    norm: mean={_pad.norm(dim=-1).mean():.4f} "
          f"std={_pad.norm(dim=-1).std():.4f}")
    # Pairwise L2 distance between consecutive pad positions
    if len(_pad) > 1:
        _diffs = (_pad[1:] - _pad[:-1]).norm(dim=-1)
        print(f"  pad consecutive L2 dist: mean={_diffs.mean():.4f} min={_diffs.min():.4f} "
              f"max={_diffs.max():.4f}")
    print()

    def _encode_lang(text):
        tokens, mask = clip_enc.encode([text])
        valid = int(mask.sum().item())
        valid = max(1, min(valid, max_lang_tokens))
        # Match the dataset's _get_lang_tokens: slice CLIP's full output,
        # DO NOT zero-fill. CLIP's pad embeddings are positionally distinct.
        out = tokens[:, :max_lang_tokens, :].float()
        return out.to(args.device), valid

    # Pick a random episode and timestep
    E = dataset.fields["observations"].shape[0]
    device = args.device

    # New layout: [action(3), proprio(3), particles(K), lang(L_lang)]
    K_particles = int(args.max_particles)
    L_lang = max_lang_tokens

    agg = {"action": [], "proprio": [], "particle": [],
           "lang_real": [], "lang_pad": []}

    for s in range(args._n_samples):
        ep = int(np.random.randint(E))
        path_len = int(dataset.path_lengths[ep])
        t0 = int(np.random.randint(max(1, path_len - args.horizon)))

        obs_t = dataset.fields.normed_observations[ep, t0]   # (K, D)
        gs_t = dataset.fields.normed_gripper_state[ep, t0] if getattr(dataset, "use_gripper_obs", False) else None
        bg_t = dataset.fields.normed_bg_features[ep, t0] if getattr(dataset, "use_bg_obs", False) else None
        lang_strs = dataset.fields.language[ep]
        lang_text = lang_strs[0] if isinstance(lang_strs, (list, tuple)) else str(lang_strs)
        print(f"\n[sample {s}] ep={ep} t={t0}/{path_len}  lang={lang_text!r}")

        parts = []
        if gs_t is not None:
            parts.append(torch.from_numpy(np.asarray(gs_t)).float().reshape(1, -1))
        if bg_t is not None:
            parts.append(torch.from_numpy(np.asarray(bg_t)).float().reshape(1, -1))
        parts.append(torch.from_numpy(np.asarray(obs_t)).float().reshape(1, -1))
        cond_0 = torch.cat(parts, dim=-1).to(device)

        lang_tokens, n_valid = _encode_lang(lang_text)
        # Build binary mask matching training: 1 for real tokens, 0 for pad.
        lang_mask_tensor = torch.zeros(1, L_lang, device=args.device)
        lang_mask_tensor[:, :n_valid] = 1.0
        print(f"[sample {s}] n_valid_lang_tokens={n_valid}/{L_lang}")

        # Build noisy x at t=0 (just zeros normalized) and call the denoiser directly
        # Follow the sampling-time shape: (batch=1, horizon, transition_dim)
        transition_dim = diffusion.transition_dim
        x = torch.randn(1, args.horizon, transition_dim, device=device)
        # apply conditioning at t=0
        x[:, 0, action_dim:] = cond_0

        # Monkey-patch CausalParticleAttention to cache attention. Tag each cached
        # call by the att_type of the module so we can separate self- from cross-
        # attention (cross-attn is the language-conditioning path).
        from diffuser.models.transformer_modules import CausalParticleAttention
        self_attn_cache = []   # list of (block_idx, attn_matrix)
        orig_forward = CausalParticleAttention.forward
        counter = {"i": 0}

        def patched_forward(self_mod, x, c=None, return_attention=False, attn_mask=None):
            y, attn = orig_forward(self_mod, x, c=c, return_attention=True, attn_mask=attn_mask)
            self_attn_cache.append((counter["i"], attn.detach()))
            counter["i"] += 1
            return y

        CausalParticleAttention.forward = patched_forward
        t = torch.zeros(1, dtype=torch.long, device=device)
        try:
            with torch.no_grad():
                _ = diffusion.model(x, None, t, return_attention=False,
                                    lang=lang_tokens, lang_mask=lang_mask_tensor)
        finally:
            CausalParticleAttention.forward = orig_forward

        # Self-attention over [action(3), proprio(3), particles(K), lang_real(n_valid), lang_pad(L-n_valid)].
        horizon = args.horizon
        for i, A in self_attn_cache:
            if A.ndim != 4:
                continue
            B, H, NT, _ = A.shape
            N = NT // horizon
            expected_N = 6 + K_particles + L_lang
            if N != expected_N:
                print(f"  [layer_{i}] WARN: expected N={expected_N}, got N={N}")
            A_mean = A.view(B, H, N, horizon, N, horizon).mean(dim=(0, 1, 3, 5)).cpu().numpy()
            row = A_mean[:3].mean(axis=0)

            action_mass = row[:3].sum()
            prop_mass   = row[3:6].sum()
            part_mass   = row[6:6 + K_particles].sum()
            lang_start  = 6 + K_particles
            lang_real   = row[lang_start : lang_start + n_valid].sum()
            lang_pad    = row[lang_start + n_valid : lang_start + L_lang].sum()
            total = row.sum()
            print(f"  [layer_{i}] "
                  f"act={100*action_mass/total:4.1f}%  "
                  f"prop={100*prop_mass/total:4.1f}%  "
                  f"part={100*part_mass/total:4.1f}%  "
                  f"lang_real={100*lang_real/total:4.1f}%  "
                  f"lang_pad={100*lang_pad/total:4.1f}%")
            agg["action"].append(action_mass)
            agg["proprio"].append(prop_mass)
            agg["particle"].append(part_mass)
            agg["lang_real"].append(lang_real)
            agg["lang_pad"].append(lang_pad)

    print("\n" + "=" * 70)
    print("AGGREGATE ACTION-QUERY SELF-ATTENTION")
    print("=" * 70)
    a_mean  = np.mean(agg["action"])
    p_mean  = np.mean(agg["proprio"])
    pa_mean = np.mean(agg["particle"])
    lr_mean = np.mean(agg["lang_real"]) if agg["lang_real"] else 0.0
    lp_mean = np.mean(agg["lang_pad"])  if agg["lang_pad"]  else 0.0
    total = a_mean + p_mean + pa_mean + lr_mean + lp_mean
    print(f"  action -> action    : {100*a_mean/total:5.2f}%")
    print(f"  action -> proprio   : {100*p_mean/total:5.2f}%")
    print(f"  action -> particles : {100*pa_mean/total:5.2f}%")
    print(f"  action -> lang_real : {100*lr_mean/total:5.2f}%")
    print(f"  action -> lang_pad  : {100*lp_mean/total:5.2f}%")

    if L_lang > 0 and n_valid > 0:
        n_pad = max(L_lang - n_valid, 0)
        # Normalize group masses to fractions of the row total so per-token rates
        # are comparable to a 1/N uniform baseline. (A_mean rows sum to 1/T, not 1,
        # because we also averaged over key-time.)
        norm_lr = lr_mean / total
        norm_lp = lp_mean / total
        real_per_tok = norm_lr / n_valid
        pad_per_tok = norm_lp / n_pad if n_pad > 0 else 0.0
        N_total = 6 + K_particles + L_lang
        uniform_per_tok = 1.0 / N_total
        ratio = real_per_tok / pad_per_tok if pad_per_tok > 0 else float("inf")
        print(f"\nPer-token attention rate (share of row, normalized):")
        print(f"  uniform baseline (1/N, N={N_total}) : {100*uniform_per_tok:5.2f}% per-token")
        print(f"  real ({n_valid:2d} tokens) : {100*real_per_tok:5.2f}% per-token  ({real_per_tok/uniform_per_tok:.2f}x uniform)")
        print(f"  pad  ({n_pad:2d} tokens) : {100*pad_per_tok:5.2f}% per-token  ({pad_per_tok/uniform_per_tok:.2f}x uniform)")
        print(f"  real / pad ratio : {ratio:.2f}x")

        # Per-layer real/pad per-token ratio — late layers often carry the
        # task-specific selection signal, so highlight the best layer.
        print("\nPer-layer real/pad per-token ratio:")
        n_layers = len(agg["lang_real"]) // max(args._n_samples, 1)
        if n_layers > 0:
            per_layer_ratio = []
            for li in range(n_layers):
                real_vals = [agg["lang_real"][s * n_layers + li]
                             for s in range(args._n_samples)]
                pad_vals = [agg["lang_pad"][s * n_layers + li]
                            for s in range(args._n_samples)]
                r_mean_l = np.mean(real_vals) / n_valid
                p_mean_l = np.mean(pad_vals) / max(n_pad, 1)
                lr = r_mean_l / p_mean_l if p_mean_l > 0 else float("inf")
                per_layer_ratio.append(lr)
                print(f"  layer_{li}: {lr:.2f}x")
            best_ratio = max(per_layer_ratio)
        else:
            best_ratio = ratio

        print("=" * 70)
        # Choose metric. If pad is masked out (pad_per_tok ~ 0), real/pad ratio
        # is vacuously infinite — fall back to comparing real per-token rate to
        # the 1/N uniform baseline. If pad has nonzero mass, use real/pad ratio.
        pad_masked = pad_per_tok < 1e-6
        real_x_uniform = real_per_tok / uniform_per_tok if uniform_per_tok > 0 else 0.0
        if pad_masked:
            print(f"(pad mass ~ 0: attention mask is active; using real × uniform as the metric)")
            if real_x_uniform > 1.5:
                print(f"[OK] Real tokens receive {real_x_uniform:.2f}x uniform attention.")
                print("     Language is being used — above-chance per-token rate.")
            elif real_x_uniform > 1.0:
                print(f"[OK] Real tokens at {real_x_uniform:.2f}x uniform (≥ flat-softmax baseline).")
                print("     Language is being read at at-least-chance per-token rate.")
            elif real_x_uniform > 0.5:
                print(f"[MARGINAL] Real at {real_x_uniform:.2f}x uniform (below chance).")
                print("     Model reads language but prefers other token groups.")
            else:
                print(f"[!] Real at {real_x_uniform:.2f}x uniform — language barely attended.")
                print("     Model mostly routes attention away from language.")
        else:
            if best_ratio > 2.0 or ratio > 1.5:
                print(f"[OK] Real tokens receive > pad per-token (best layer {best_ratio:.2f}x).")
                print("     Language is being used to select content.")
            elif best_ratio > 1.3 or ratio > 1.15:
                print(f"[WEAK] Mild preference for real over pad (best layer {best_ratio:.2f}x).")
                print("     Language signal is present but not strong.")
            elif best_ratio > 1.05:
                print(f"[MARGINAL] Real barely beats pad per-token (best layer {best_ratio:.2f}x).")
                print("     Model is not distinguishing content from padding.")
            else:
                print(f"[!] Real and pad are equivalent per-token (best layer {best_ratio:.2f}x).")
                print("     Model is treating lang positions as undifferentiated noise.")
    else:
        print("\n(skipping real/pad comparison — no valid lang tokens)")
    print("=" * 70)


if __name__ == "__main__":
    main(sys.argv[1:])
