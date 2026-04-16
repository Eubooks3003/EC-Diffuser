"""
AdaLNPINTDenoiser
-----------------

Denoising transformer over (action, proprioception, particles) trajectories.

When gripper_dim > 0, action and proprioception are each split into three
semantically distinct tokens — position, rotation, gripper open/close — giving
six robot tokens in the transformer sequence alongside the particle tokens:

    tokens = [a_pos, a_rot, a_grip, p_pos, p_rot, p_grip, particle_1..K]

Default component sizes (single-arm Panda OSC_POSE, gripper_state format
[pos(3), rot6d(6), open(1)]):
    action_dim=7:  act_pos_dim=3, act_rot_dim=3, act_grip_dim=1
    gripper_dim=10: prop_pos_dim=3, prop_rot_dim=6, prop_grip_dim=1

The flat input/output tensor layout is unchanged:
    x: [batch_size, T, action_dim + gripper_dim + bg_dim + (n_particles * features_dim)]

When gripper_dim == 0 the legacy single-action-token path is used unchanged.
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
                 prop_pos_dim=3, prop_rot_dim=6, prop_grip_dim=1, **kwargs):
        super(AdaLNPINTDenoiser, self).__init__()

        self.features_dim = features_dim
        self.action_dim = action_dim
        self.gripper_dim = gripper_dim
        self.bg_dim = bg_dim
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
        self.split_robot_tokens = gripper_dim > 0
        if self.split_robot_tokens:
            assert act_pos_dim + act_rot_dim + act_grip_dim == action_dim, (
                f"action sub-dims {act_pos_dim}+{act_rot_dim}+{act_grip_dim} "
                f"must sum to action_dim={action_dim}")
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

        if self.split_robot_tokens:
            # Six robot tokens: action(pos/rot/grip) + proprio(pos/rot/grip).
            self.a_pos_projection = _make_proj(act_pos_dim)
            self.a_rot_projection = _make_proj(act_rot_dim)
            self.a_grip_projection = _make_proj(act_grip_dim)
            self.p_pos_projection = _make_proj(prop_pos_dim)
            self.p_rot_projection = _make_proj(prop_rot_dim)
            self.p_grip_projection = _make_proj(prop_grip_dim)

            self.a_pos_encoding = nn.Parameter(0.02 * torch.randn(1, 1, projection_dim))
            self.a_rot_encoding = nn.Parameter(0.02 * torch.randn(1, 1, projection_dim))
            self.a_grip_encoding = nn.Parameter(0.02 * torch.randn(1, 1, projection_dim))
            self.p_pos_encoding = nn.Parameter(0.02 * torch.randn(1, 1, projection_dim))
            self.p_rot_encoding = nn.Parameter(0.02 * torch.randn(1, 1, projection_dim))
            self.p_grip_encoding = nn.Parameter(0.02 * torch.randn(1, 1, projection_dim))
        else:
            # Legacy single action token (no proprio).
            self.action_projection = _make_proj(action_dim)

        # Instantiate the AdaLN Particle Transformer.
        self.particle_transformer = AdaLNParticleTransformer(
            self.projection_dim, n_head, n_layer, block_size, self.projection_dim,
            attn_pdrop=dropout, resid_pdrop=dropout,
            hidden_dim_multiplier=4,
            positional_bias=positional_bias,
            activation='gelu', max_particles=max_particles
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

        if self.split_robot_tokens:
            self.a_pos_decoder = _make_dec(act_pos_dim)
            self.a_rot_decoder = _make_dec(act_rot_dim)
            self.a_grip_decoder = _make_dec(act_grip_dim)
            self.p_pos_decoder = _make_dec(prop_pos_dim)
            self.p_rot_decoder = _make_dec(prop_rot_dim)
            self.p_grip_decoder = _make_dec(prop_grip_dim)
        else:
            self.action_decoder = _make_dec(action_dim)
            self.action_encoding = nn.Parameter(0.02 * torch.randn(1, 1, projection_dim))

        # Particle encoding: either shared or view-specific for multi-view inputs.
        if self.multiview:
            self.view1_encoding = nn.Parameter(0.02 * torch.randn(1, 1, 1, projection_dim))
            self.view2_encoding = nn.Parameter(0.02 * torch.randn(1, 1, 1, projection_dim))
        else:
            self.particle_encoding = nn.Parameter(0.02 * torch.randn(1, 1, 1, projection_dim))

    def forward(self, x, cond, time, return_attention=False):
        """
        Input/output flat layout (both paths):
            [action(action_dim), gripper(gripper_dim), bg(bg_dim), particles(K*features_dim)]

        When gripper_dim > 0, action and gripper are each split into three
        sub-components (pos, rot, grip) and enter the transformer as six
        separate tokens followed by the K particle tokens. The output is
        reassembled into the original flat layout.
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

        if self.split_robot_tokens:
            # Slice action into (pos, rot, grip) and gripper_state into (pos, rot, grip).
            ap0 = 0
            ap1 = ap0 + self.act_pos_dim
            ar1 = ap1 + self.act_rot_dim
            ag1 = ar1 + self.act_grip_dim  # == action_dim

            pp0 = self.action_dim
            pp1 = pp0 + self.prop_pos_dim
            pr1 = pp1 + self.prop_rot_dim
            pg1 = pr1 + self.prop_grip_dim  # == action_dim + gripper_dim

            a_pos = x[:, :, ap0:ap1]
            a_rot = x[:, :, ap1:ar1]
            a_grip = x[:, :, ar1:ag1]
            p_pos = x[:, :, pp0:pp1]
            p_rot = x[:, :, pp1:pr1]
            p_grip = x[:, :, pr1:pg1]

            a_pos_tok = self.a_pos_projection(a_pos) + self.a_pos_encoding.repeat(bs, T, 1)
            a_rot_tok = self.a_rot_projection(a_rot) + self.a_rot_encoding.repeat(bs, T, 1)
            a_grip_tok = self.a_grip_projection(a_grip) + self.a_grip_encoding.repeat(bs, T, 1)
            p_pos_tok = self.p_pos_projection(p_pos) + self.p_pos_encoding.repeat(bs, T, 1)
            p_rot_tok = self.p_rot_projection(p_rot) + self.p_rot_encoding.repeat(bs, T, 1)
            p_grip_tok = self.p_grip_projection(p_grip) + self.p_grip_encoding.repeat(bs, T, 1)

            # Transformer sequence: six robot tokens then particle tokens.
            x_cat = torch.cat([
                a_pos_tok.unsqueeze(2),
                a_rot_tok.unsqueeze(2),
                a_grip_tok.unsqueeze(2),
                p_pos_tok.unsqueeze(2),
                p_rot_tok.unsqueeze(2),
                p_grip_tok.unsqueeze(2),
                new_state_particles,
            ], dim=2)
            particle_start_token_idx = 6
            anchor = a_pos_tok  # used by the transformer's AdaLN anchor path
        else:
            actions = x[:, :, :self.action_dim]
            action_particle = self.action_projection(actions) + self.action_encoding.repeat(bs, T, 1)
            x_cat = torch.cat([action_particle.unsqueeze(2), new_state_particles], dim=2)
            particle_start_token_idx = 1
            anchor = action_particle

        # Add time embedding to every token and permute to [bs, n_tokens, T, projection_dim].
        x_proj = x_cat + t_embed[:, None, None, :]
        x_proj = x_proj.permute(0, 2, 1, 3)

        # ---------------------------------------------------------------------
        # Apply the particle transformer.
        if return_attention:
            particles_trans, attention_dict = self.particle_transformer(
                x_proj, anchor, t_embed, return_attention=return_attention)
        else:
            particles_trans = self.particle_transformer(x_proj, anchor, t_embed)
        particles_trans = particles_trans.permute(0, 2, 1, 3)  # [bs, T, n_tokens, projection_dim]

        # ---------------------------------------------------------------------
        # Decode.
        particle_decoder_out = self.particle_decoder(particles_trans[:, :, particle_start_token_idx:, :])
        particle_decoder_out = particle_decoder_out.view(bs, T, -1)

        if self.split_robot_tokens:
            a_pos_out = self.a_pos_decoder(particles_trans[:, :, 0, :])
            a_rot_out = self.a_rot_decoder(particles_trans[:, :, 1, :])
            a_grip_out = self.a_grip_decoder(particles_trans[:, :, 2, :])
            p_pos_out = self.p_pos_decoder(particles_trans[:, :, 3, :])
            p_rot_out = self.p_rot_decoder(particles_trans[:, :, 4, :])
            p_grip_out = self.p_grip_decoder(particles_trans[:, :, 5, :])

            parts = [a_pos_out, a_rot_out, a_grip_out,
                     p_pos_out, p_rot_out, p_grip_out]
            if self.bg_dim > 0 and bg_features is not None:
                parts.append(bg_features)
            parts.append(particle_decoder_out)
            x_out = torch.cat(parts, dim=-1)
        else:
            action_decoder_out = self.action_decoder(particles_trans[:, :, 0, :])
            parts = [action_decoder_out]
            if self.bg_dim > 0 and bg_features is not None:
                parts.append(bg_features)
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
