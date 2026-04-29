"""
AdaLNPINTDenoiser
-----------------

Denoising transformer over (action, proprioception, particles) trajectories.

Two independent flags control robot-token construction:
  * split_action_tokens: if True, the action is split into three semantic
    sub-tokens (pos, rot, grip) with separate projections and decoder heads.
    If False, the action is a single monolithic token.
  * gripper_dim > 0: if True, proprioception is included as three sub-tokens
    (pos, rot, grip). If 0, no proprio token is added.

These two axes are independent, giving four valid modes. Transformer sequence:

    tokens = [action_tokens..., proprio_tokens..., (bg?), particle_1..K]

where action_tokens is 1 or 3 tokens and proprio_tokens is 0 or 3 tokens.

Default component sizes (single-arm Panda OSC_POSE, gripper_state format
[pos(3), rot6d(6), open(1)]):
    action_dim=7:  act_pos_dim=3, act_rot_dim=3, act_grip_dim=1
    gripper_dim=10: prop_pos_dim=3, prop_rot_dim=6, prop_grip_dim=1

The flat input/output tensor layout is unchanged:
    x: [batch_size, T, action_dim + gripper_dim + bg_dim + (n_particles * features_dim)]

If split_action_tokens is left at its default (None), it is derived as
`gripper_dim > 0` to preserve the previous coupled behavior.
"""

import torch
from torch import nn
from diffuser.models.transformer_modules import (
    AdaLNParticleTransformer,
    SinusoidalPosEmb,
    RandomOrLearnedSinusoidalPosEmb,
)

class AdaLNPINTDenoiser(nn.Module):
    """
    AdaLNPINTDenoiser

    Implements a denoising model based on an Adaptive Layer Normalized Particle Interaction
    Transformer. It processes sequences of particle state features concatenated with action
    information and conditioned on a time signal.

    Parameters:
        features_dim (int): Dimensionality of each particle's feature vector.
        action_dim (int): Dimensionality of the action vector.
        hidden_dim (int): Hidden dimension used in projection layers.
        projection_dim (int): Dimension of the latent space in the transformer.
        n_head (int): Number of attention heads in the transformer.
        n_layer (int): Number of transformer layers.
        block_size (int): Time horizon (number of time steps).
        dropout (float): Dropout probability for transformer components.
        predict_delta (bool): If True, the model predicts a delta change rather than an absolute value.
        positional_bias (bool): If True, applies positional bias in the transformer.
        max_particles (int or None): Maximum number of particles (for relative positional bias).
        learned_sinusoidal_cond (bool): If True, use a learned sinusoidal embedding for time conditioning.
        random_fourier_features (bool): If True, use fixed random Fourier features.
        learned_sinusoidal_dim (int): Dimensionality for the learned sinusoidal (or Fourier) features.
        multiview (bool): If True, use separate encodings for multi-view particle inputs.
        gripper_dim (int): Dimensionality of gripper state. If > 0, gripper is treated as an
            additional token. Typical format: [pos(3), rot_6d(6), open(1)] = 10 dims.
    """
    def __init__(self, features_dim=2, action_dim=3, hidden_dim=256, projection_dim=256,
                 n_head=8, n_layer=6, block_size=50, dropout=0.1,
                 predict_delta=False, positional_bias=True, max_particles=4,
                 learned_sinusoidal_cond=False, random_fourier_features=False,
                 learned_sinusoidal_dim=16, multiview=False, gripper_dim=0, bg_dim=0,
                 act_pos_dim=3, act_rot_dim=3, act_grip_dim=1,
                 prop_pos_dim=3, prop_rot_dim=6, prop_grip_dim=1,
                 split_action_tokens=None,
                 lang_dim=0, use_cond_tokens=False, **kwargs):
        super(AdaLNPINTDenoiser, self).__init__()

        self.features_dim = features_dim
        self.action_dim = action_dim
        self.gripper_dim = gripper_dim
        self.bg_dim = bg_dim
        self.lang_dim = lang_dim  # CLIP hidden size (e.g. 512); 0 disables language conditioning
        self.use_cond_tokens = use_cond_tokens  # 3DDA-style: current-keypose cond as separate context tokens
        self.predict_delta = predict_delta
        self.projection_dim = projection_dim
        self.max_particles = max_particles
        self.multiview = multiview

        self.act_pos_dim = act_pos_dim
        self.act_rot_dim = act_rot_dim
        self.act_grip_dim = act_grip_dim
        self.prop_pos_dim = prop_pos_dim
        self.prop_rot_dim = prop_rot_dim
        self.prop_grip_dim = prop_grip_dim
        # Default to the previous coupled behavior so existing configs that
        # only set gripper_dim keep working unchanged.
        if split_action_tokens is None:
            split_action_tokens = gripper_dim > 0
        self.split_action_tokens = bool(split_action_tokens)
        self.use_proprio = gripper_dim > 0
        if self.split_action_tokens:
            assert act_pos_dim + act_rot_dim + act_grip_dim == action_dim, (
                f"action sub-dims {act_pos_dim}+{act_rot_dim}+{act_grip_dim} "
                f"must sum to action_dim={action_dim}")
        if self.use_proprio:
            assert prop_pos_dim + prop_rot_dim + prop_grip_dim == gripper_dim, (
                f"proprio sub-dims {prop_pos_dim}+{prop_rot_dim}+{prop_grip_dim} "
                f"must sum to gripper_dim={gripper_dim}")
        # block_size is the time horizon

        # Define an intermediate time embedding dimension.
        time_dim = projection_dim * 4

        # Decide whether to use random/learned Fourier features for time conditioning.
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            # Fourier feature output is concatenated with the original scalar, so add 1.
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(projection_dim)
            fourier_dim = projection_dim

        # Time embedding network.
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, projection_dim)
        )

        # Particle feature projection network.
        self.particle_projection = nn.Sequential(
            nn.Linear(self.features_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.projection_dim)
        )

        def _make_proj(in_dim):
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.projection_dim),
            )

        # Action side: 1 token (single head) or 3 tokens (pos/rot/grip heads).
        if self.split_action_tokens:
            self.a_pos_projection = _make_proj(act_pos_dim)
            self.a_rot_projection = _make_proj(act_rot_dim)
            self.a_grip_projection = _make_proj(act_grip_dim)
            self.a_pos_encoding = nn.Parameter(0.02 * torch.randn(1, 1, projection_dim))
            self.a_rot_encoding = nn.Parameter(0.02 * torch.randn(1, 1, projection_dim))
            self.a_grip_encoding = nn.Parameter(0.02 * torch.randn(1, 1, projection_dim))
        else:
            self.action_projection = _make_proj(action_dim)
            self.action_encoding = nn.Parameter(0.02 * torch.randn(1, 1, projection_dim))

        # Proprio side: 0 tokens or 3 tokens (pos/rot/grip).
        if self.use_proprio:
            self.p_pos_projection = _make_proj(prop_pos_dim)
            self.p_rot_projection = _make_proj(prop_rot_dim)
            self.p_grip_projection = _make_proj(prop_grip_dim)
            self.p_pos_encoding = nn.Parameter(0.02 * torch.randn(1, 1, projection_dim))
            self.p_rot_encoding = nn.Parameter(0.02 * torch.randn(1, 1, projection_dim))
            self.p_grip_encoding = nn.Parameter(0.02 * torch.randn(1, 1, projection_dim))

        # Language tokens are concatenated into the self-attention sequence with a
        # shared type encoding, matching object-centric-gauss-splat's
        # AdaLNPINTDenoiserSelfAtten (ParticleSplat reference).
        if self.lang_dim > 0:
            self.lang_projection = nn.Sequential(
                nn.Linear(self.lang_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.projection_dim),
            )
            self.lang_encoding = nn.Parameter(0.02 * torch.randn(1, 1, 1, projection_dim))

        self.particle_transformer = AdaLNParticleTransformer(
            self.projection_dim, n_head, n_layer, block_size, self.projection_dim,
            attn_pdrop=dropout, resid_pdrop=dropout,
            hidden_dim_multiplier=4,
            positional_bias=positional_bias,
            activation='gelu', max_particles=max_particles,
        )

        # Decoder networks for particle outputs.
        self.particle_decoder = nn.Sequential(
            nn.Linear(self.projection_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.features_dim)
        )

        def _make_dec(out_dim):
            return nn.Sequential(
                nn.Linear(self.projection_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
            )

        if self.split_action_tokens:
            self.a_pos_decoder = _make_dec(act_pos_dim)
            self.a_rot_decoder = _make_dec(act_rot_dim)
            self.a_grip_decoder = _make_dec(act_grip_dim)
        else:
            self.action_decoder = _make_dec(action_dim)

        if self.use_proprio:
            self.p_pos_decoder = _make_dec(prop_pos_dim)
            self.p_rot_decoder = _make_dec(prop_rot_dim)
            self.p_grip_decoder = _make_dec(prop_grip_dim)

        # Particle encoding: either shared or view-specific for multi-view inputs.
        if self.multiview:
            self.view1_encoding = nn.Parameter(0.02 * torch.randn(1, 1, 1, projection_dim))
            self.view2_encoding = nn.Parameter(0.02 * torch.randn(1, 1, 1, projection_dim))
        else:
            self.particle_encoding = nn.Parameter(0.02 * torch.randn(1, 1, 1, projection_dim))

        # Background features: prior versions sliced bg from input and concat'd
        # it straight to the output unchanged, so bg_dim was never denoised.
        # Project it into a token, run through the transformer, decode back.
        if self.bg_dim > 0:
            self.bg_projection = _make_proj(self.bg_dim)
            self.bg_decoder = _make_dec(self.bg_dim)
            self.bg_encoding = nn.Parameter(0.02 * torch.randn(1, 1, projection_dim))

        # Conditioning-token encodings: 3DDA-style separate context tokens for the
        # current keypose. Distinct from trajectory encodings so the model can
        # tell "current scene context" from "next-keypose prediction." Reuse
        # projection layers (p_pos/p_rot/p_grip/bg/particle) since modalities
        # are identical to trajectory tokens.
        if self.use_cond_tokens:
            if self.use_proprio:
                self.cond_p_pos_encoding = nn.Parameter(0.02 * torch.randn(1, 1, projection_dim))
                self.cond_p_rot_encoding = nn.Parameter(0.02 * torch.randn(1, 1, projection_dim))
                self.cond_p_grip_encoding = nn.Parameter(0.02 * torch.randn(1, 1, projection_dim))
            if self.bg_dim > 0:
                self.cond_bg_encoding = nn.Parameter(0.02 * torch.randn(1, 1, projection_dim))
            if self.multiview:
                self.cond_view1_encoding = nn.Parameter(0.02 * torch.randn(1, 1, projection_dim))
                self.cond_view2_encoding = nn.Parameter(0.02 * torch.randn(1, 1, projection_dim))
            else:
                self.cond_particle_encoding = nn.Parameter(0.02 * torch.randn(1, 1, projection_dim))

    def forward(self, x, cond, time, return_attention=False, lang=None, lang_mask=None):
        """
        Input/output flat layout:
            [action(action_dim), gripper(gripper_dim), bg(bg_dim), particles(K*features_dim)]

        Action enters the transformer as 1 token (single head) or 3 tokens
        (pos/rot/grip heads), controlled by split_action_tokens. Proprio
        enters as 0 or 3 tokens, controlled by gripper_dim > 0. The output
        is reassembled back into the flat layout.
        """
        # ---------------------------------------------------------------------
        # Flat input layout: [action(action_dim), gripper(gripper_dim), bg(bg_dim), particles]
        bs, T, f = x.size()

        # Slice the bg and particle regions (common to both paths).
        bg_start = self.action_dim + self.gripper_dim
        particle_start_idx = bg_start + self.bg_dim
        if self.bg_dim > 0:
            bg_features = x[:, :, bg_start:particle_start_idx]
        else:
            bg_features = None
        x_particles = x[:, :, particle_start_idx:].view(bs, T, -1, self.features_dim)

        # Project particles (with optional per-view encoding).
        state_particles = self.particle_projection(x_particles)
        if self.multiview:
            n_particles = state_particles.size(2) // 2
            particles_view1 = state_particles[:, :, :n_particles, :] + self.view1_encoding.repeat(bs, T, n_particles, 1)
            particles_view2 = state_particles[:, :, n_particles:, :] + self.view2_encoding.repeat(bs, T, n_particles, 1)
            new_state_particles = torch.cat([particles_view1, particles_view2], dim=2)
        else:
            new_state_particles = state_particles + self.particle_encoding.repeat(bs, T, state_particles.size(2), 1)

        t_embed = self.time_mlp(time)  # [bs, projection_dim]

        # Build bg token (inserted between robot tokens and particle tokens).
        bg_tok_list = []
        if self.bg_dim > 0 and bg_features is not None:
            bg_tok = self.bg_projection(bg_features) + self.bg_encoding  # (bs, T, projection_dim)
            bg_tok_list = [bg_tok.unsqueeze(2)]

        # Action tokens: 1 (single head) or 3 (pos/rot/grip heads).
        action_slice = x[:, :, :self.action_dim]
        if self.split_action_tokens:
            ap1 = self.act_pos_dim
            ar1 = ap1 + self.act_rot_dim
            ag1 = ar1 + self.act_grip_dim  # == action_dim
            a_pos_tok = self.a_pos_projection(action_slice[:, :, :ap1]) + self.a_pos_encoding.repeat(bs, T, 1)
            a_rot_tok = self.a_rot_projection(action_slice[:, :, ap1:ar1]) + self.a_rot_encoding.repeat(bs, T, 1)
            a_grip_tok = self.a_grip_projection(action_slice[:, :, ar1:ag1]) + self.a_grip_encoding.repeat(bs, T, 1)
            action_tok_list = [a_pos_tok.unsqueeze(2), a_rot_tok.unsqueeze(2), a_grip_tok.unsqueeze(2)]
            anchor = a_pos_tok  # used by the transformer's AdaLN anchor path
        else:
            action_particle = self.action_projection(action_slice) + self.action_encoding.repeat(bs, T, 1)
            action_tok_list = [action_particle.unsqueeze(2)]
            anchor = action_particle

        # Proprio tokens: 0 or 3 (pos/rot/grip).
        proprio_tok_list = []
        if self.use_proprio:
            pp0 = self.action_dim
            pp1 = pp0 + self.prop_pos_dim
            pr1 = pp1 + self.prop_rot_dim
            pg1 = pr1 + self.prop_grip_dim  # == action_dim + gripper_dim
            p_pos_tok = self.p_pos_projection(x[:, :, pp0:pp1]) + self.p_pos_encoding.repeat(bs, T, 1)
            p_rot_tok = self.p_rot_projection(x[:, :, pp1:pr1]) + self.p_rot_encoding.repeat(bs, T, 1)
            p_grip_tok = self.p_grip_projection(x[:, :, pr1:pg1]) + self.p_grip_encoding.repeat(bs, T, 1)
            proprio_tok_list = [p_pos_tok.unsqueeze(2), p_rot_tok.unsqueeze(2), p_grip_tok.unsqueeze(2)]

        n_action_toks = len(action_tok_list)
        n_proprio_toks = len(proprio_tok_list)

        # ---------------------------------------------------------------------
        # Conditioning tokens: 3DDA-style separate context for the *current*
        # keypose. cond[0] = concat(gripper, bg, particles) at the conditioning
        # frame; project each modality through the existing trajectory
        # projections (weight reuse), tag with cond-specific encodings, then
        # broadcast across T (time-invariant context). Inserted between bg and
        # particles so particle_start_token_idx shifts cleanly and the existing
        # particle decoder slice [particle_start_token_idx:] stays correct.
        n_cond = 0
        cond_tokens_for_cat = None
        if self.use_cond_tokens and cond is not None and 0 in cond:
            cond_vec = cond[0]  # [bs, gripper_dim + bg_dim + n_particles*features_dim]
            cond_pos_offset = 0
            cond_token_list = []

            # Proprio cond (3 sub-tokens) -- mirrors trajectory proprio split.
            if self.use_proprio:
                cg = cond_vec[:, cond_pos_offset:cond_pos_offset + self.gripper_dim]
                cond_pos_offset += self.gripper_dim
                cpp1 = self.prop_pos_dim
                cpr1 = cpp1 + self.prop_rot_dim
                cpg1 = cpr1 + self.prop_grip_dim
                cp_pos = self.p_pos_projection(cg[:, :cpp1]) + self.cond_p_pos_encoding[:, 0]
                cp_rot = self.p_rot_projection(cg[:, cpp1:cpr1]) + self.cond_p_rot_encoding[:, 0]
                cp_grip = self.p_grip_projection(cg[:, cpr1:cpg1]) + self.cond_p_grip_encoding[:, 0]
                cond_token_list.extend([
                    cp_pos.unsqueeze(1), cp_rot.unsqueeze(1), cp_grip.unsqueeze(1),
                ])  # each [bs, 1, projection_dim]

            # Bg cond.
            if self.bg_dim > 0:
                cbg = cond_vec[:, cond_pos_offset:cond_pos_offset + self.bg_dim]
                cond_pos_offset += self.bg_dim
                cbg_tok = self.bg_projection(cbg) + self.cond_bg_encoding[:, 0]
                cond_token_list.append(cbg_tok.unsqueeze(1))

            # Particles cond.
            cond_part_flat = cond_vec[:, cond_pos_offset:]
            cond_particles_in = cond_part_flat.view(bs, -1, self.features_dim)
            cond_part_emb = self.particle_projection(cond_particles_in)  # [bs, n_part, proj_dim]
            if self.multiview:
                n_p = cond_part_emb.size(1) // 2
                cv1 = cond_part_emb[:, :n_p] + self.cond_view1_encoding[:, 0]
                cv2 = cond_part_emb[:, n_p:] + self.cond_view2_encoding[:, 0]
                cond_part_emb = torch.cat([cv1, cv2], dim=1)
            else:
                cond_part_emb = cond_part_emb + self.cond_particle_encoding[:, 0]
            cond_token_list.append(cond_part_emb)

            cond_tokens_2d = torch.cat(cond_token_list, dim=1)  # [bs, n_cond, projection_dim]
            n_cond = cond_tokens_2d.size(1)
            cond_tokens_for_cat = cond_tokens_2d.unsqueeze(1).expand(-1, T, -1, -1)  # [bs, T, n_cond, projection_dim]

        # Transformer sequence: [action tokens, proprio tokens, (bg?), (cond?), particles].
        cat_list = [*action_tok_list, *proprio_tok_list, *bg_tok_list]
        if cond_tokens_for_cat is not None:
            cat_list.append(cond_tokens_for_cat)
        cat_list.append(new_state_particles)
        x_cat = torch.cat(cat_list, dim=2)
        bg_token_idx = n_action_toks + n_proprio_toks
        particle_start_token_idx = bg_token_idx + len(bg_tok_list) + n_cond

        # Language conditioning: concatenate projected tokens into the self-attention
        # sequence with shared type encoding. Time embedding added uniformly below.
        n_lang = 0
        lang_valid = None
        if self.lang_dim > 0 and lang is not None:
            L_lang = lang.size(1)
            lang_tok = self.lang_projection(lang)
            lang_tok = lang_tok.unsqueeze(1).expand(bs, T, L_lang, self.projection_dim)
            lang_tok = lang_tok + self.lang_encoding
            x_cat = torch.cat([x_cat, lang_tok], dim=2)
            n_lang = L_lang
            lang_valid = lang_mask

        # Padding mask: 1 for robot/particle/valid-lang tokens, 0 for lang pad.
        attn_mask = None
        if n_lang > 0 and lang_valid is not None:
            n_nonlang = x_cat.size(2) - n_lang
            ones = torch.ones(bs, n_nonlang, device=x.device, dtype=lang_valid.dtype)
            attn_mask = torch.cat([ones, lang_valid], dim=1)

        x_proj = x_cat + t_embed[:, None, None, :]
        x_proj = x_proj.permute(0, 2, 1, 3)

        # ---------------------------------------------------------------------
        # Apply the particle transformer.
        if return_attention:
            particles_trans, attention_dict = self.particle_transformer(
                x_proj, anchor, t_embed, return_attention=return_attention, attn_mask=attn_mask)
        else:
            particles_trans = self.particle_transformer(x_proj, anchor, t_embed, attn_mask=attn_mask)
        particles_trans = particles_trans.permute(0, 2, 1, 3)  # [bs, T, n_tokens, projection_dim]

        if n_lang > 0:
            particles_trans = particles_trans[:, :, :particles_trans.size(2) - n_lang, :]

        # ---------------------------------------------------------------------
        # Decode.
        particle_decoder_out = self.particle_decoder(particles_trans[:, :, particle_start_token_idx:, :])
        particle_decoder_out = particle_decoder_out.view(bs, T, -1)

        # Decode the bg token output (if enabled).
        bg_decoder_out = None
        if self.bg_dim > 0 and bg_features is not None:
            bg_decoder_out = self.bg_decoder(particles_trans[:, :, bg_token_idx, :])

        parts = []
        if self.split_action_tokens:
            parts.append(self.a_pos_decoder(particles_trans[:, :, 0, :]))
            parts.append(self.a_rot_decoder(particles_trans[:, :, 1, :]))
            parts.append(self.a_grip_decoder(particles_trans[:, :, 2, :]))
        else:
            parts.append(self.action_decoder(particles_trans[:, :, 0, :]))

        if self.use_proprio:
            parts.append(self.p_pos_decoder(particles_trans[:, :, n_action_toks + 0, :]))
            parts.append(self.p_rot_decoder(particles_trans[:, :, n_action_toks + 1, :]))
            parts.append(self.p_grip_decoder(particles_trans[:, :, n_action_toks + 2, :]))

        if bg_decoder_out is not None:
            parts.append(bg_decoder_out)
        parts.append(particle_decoder_out)
        x_out = torch.cat(parts, dim=-1)

        if return_attention:
            return x_out, attention_dict
        else:
            return x_out

# ------------------------------------------------------------------------------
# Test block
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    batch_size = 32
    timessteps = 5

    # Test without gripper token
    print("=" * 60)
    print("Test 1: Without gripper token (gripper_dim=0)")
    print("=" * 60)
    model = AdaLNPINTDenoiser(features_dim=10, action_dim=3, hidden_dim=256, projection_dim=256,
                        n_head=8, n_layer=6, block_size=timessteps, dropout=0.1,
                        predict_delta=False, positional_bias=False, max_particles=None,
                        learned_sinusoidal_cond=False, random_fourier_features=False, learned_sinusoidal_dim=16,
                        gripper_dim=0)
    in_particles = torch.randn(batch_size, timessteps, 240)
    actions = torch.randn(batch_size, timessteps, 3)
    t = torch.randint(0, 1000, (batch_size,), device=in_particles.device).long()

    # Concatenate actions and particle features.
    x = torch.cat([actions, in_particles], dim=-1)
    model_out = model(x, cond=None, time=t, return_attention=False)
    print("Input shape:", x.shape)
    print("Output shape:", model_out.shape)
    assert model_out.shape == x.shape, "Output shape should match input shape"

    # Test split-token path (action=7D OSC_POSE, gripper=10D proprio)
    print("\n" + "=" * 60)
    print("Test 2: Split robot tokens (action_dim=7, gripper_dim=10)")
    print("=" * 60)
    action_dim_split = 7   # Δpos(3) + Δaxis_angle(3) + grip_cmd(1)
    gripper_dim = 10       # eef_pos(3) + rot6d(6) + grip_open(1)
    model_with_gripper = AdaLNPINTDenoiser(
        features_dim=10, action_dim=action_dim_split, hidden_dim=256, projection_dim=256,
        n_head=8, n_layer=6, block_size=timessteps, dropout=0.1,
        predict_delta=False, positional_bias=False, max_particles=None,
        learned_sinusoidal_cond=False, random_fourier_features=False, learned_sinusoidal_dim=16,
        gripper_dim=gripper_dim
    )
    actions_split = torch.randn(batch_size, timessteps, action_dim_split)
    gripper_state = torch.randn(batch_size, timessteps, gripper_dim)
    x_with_gripper = torch.cat([actions_split, gripper_state, in_particles], dim=-1)
    model_out_with_gripper = model_with_gripper(x_with_gripper, cond=None, time=t, return_attention=False)
    print("Input shape:", x_with_gripper.shape)
    print("Output shape:", model_out_with_gripper.shape)
    assert model_out_with_gripper.shape == x_with_gripper.shape, "Output shape should match input shape"

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
