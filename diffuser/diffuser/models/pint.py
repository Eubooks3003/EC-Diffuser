"""
AdaLNPINTDenoiser
-----------------

Denoising transformer over (action, proprioception, particles) trajectories.

Action and proprio tokenization are now decoupled by `split_action_tokens`
and the implicit `has_proprio = gripper_dim > 0`. Robot tokens are built
piecewise: action side first, then proprio side (if any), then particles.

    split_action_tokens=True,  has_proprio=True   → 6 tokens:
        [a_pos, a_rot, a_grip, p_pos, p_rot, p_grip, particle_1..K]
    split_action_tokens=True,  has_proprio=False  → 3 tokens:
        [a_pos, a_rot, a_grip, particle_1..K]
    split_action_tokens=False, has_proprio=True   → 4 tokens:
        [action,                p_pos, p_rot, p_grip, particle_1..K]
    split_action_tokens=False, has_proprio=False  → 1 token (legacy):
        [action,                                       particle_1..K]

`split_action_tokens=None` (default) preserves the historical behaviour:
auto-derive to `gripper_dim > 0`, so old configs/ckpts load unchanged.

Auxiliary action tokenizations (`aux_action_token_groups`)
----------------------------------------------------------
The action slice can additionally be re-tokenized by one or more *auxiliary*
branches, each with its own projections / type-encodings / decoder heads.
Every branch reads the SAME `x[:, :, :action_dim]` values as the primary
branch — there is no second action slice and no second noise draw, so the
diffusion corruption is shared across representations by construction.

    action_token_groups=[3,3,1], aux_action_token_groups=[[7]]  → 4 action tokens:
        [a_pos, a_rot, a_grip, p_pos, p_rot, p_grip, AUX(a_all), particle_1..K]

Auxiliary tokens are appended after the proprio tokens (i.e. immediately
before the particles) so primary token indices are identical to a run
without them. They always participate in attention — including at sampling
time — but their decoders only run when `return_aux=True`, and their output
never enters `x_out`. The executed action is always the primary decode.

NOTE: token *ordering* is only irrelevant because `positional_bias=False` in
these configs. With positional_bias enabled, the relative-position bias is
indexed over the token axis and `max_particles` bounds the TOTAL token count.

Default component sizes (single-arm Panda OSC_POSE, gripper_state format
[pos(3), rot6d(6), open(1)]):
    action_dim=7:  act_pos_dim=3, act_rot_dim=3, act_grip_dim=1
    gripper_dim=10: prop_pos_dim=3, prop_rot_dim=6, prop_grip_dim=1

The flat input/output tensor layout is unchanged regardless of the flags:
    x: [batch_size, T, action_dim + gripper_dim + bg_dim + (n_particles * features_dim)]
"""

import torch
from torch import nn
from diffuser.models.transformer_modules import (
    AdaLNParticleTransformer,
    SinusoidalPosEmb,
    RandomOrLearnedSinusoidalPosEmb,
)

def _normalize_aux_action_groups(spec, action_dim):
    """Normalize `aux_action_token_groups` into a list of branches.

    Each branch is a list of sub-dims summing to `action_dim`. Accepted forms:
        None            -> []                    (no auxiliary branches)
        'per_dim'       -> [[1]*action_dim]      (one uniform branch)
        [7]             -> [[7]]                 (flat list = a single branch)
        [[7]]           -> [[7]]
        [[3,3,1], 'per_dim'] -> two branches
    """
    if spec is None:
        return []
    if isinstance(spec, str):
        spec = [spec]
    elif len(spec) == 0:
        return []
    elif all(isinstance(g, (int, float)) for g in spec):
        spec = [spec]  # flat list of sub-dims == one branch

    branches = []
    for branch in spec:
        if branch == 'per_dim':
            branch = [1] * action_dim
        branch = [int(d) for d in branch]
        assert sum(branch) == action_dim, (
            f"aux action branch {branch} must sum to action_dim={action_dim}")
        branches.append(branch)
    return branches


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
                 n_tasks=1, split_action_tokens=None,
                 action_token_groups=None, proprio_token_groups=None,
                 aux_action_token_groups=None, **kwargs):
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
        # Back-compat: when caller does not specify, auto-derive to the
        # historical `gripper_dim > 0` semantics so old ckpts/configs load.
        if split_action_tokens is None:
            self.split_action_tokens = gripper_dim > 0
        else:
            self.split_action_tokens = bool(split_action_tokens)
        self.has_proprio = gripper_dim > 0

        if self.split_action_tokens:
            assert act_pos_dim + act_rot_dim + act_grip_dim == action_dim, (
                f"action sub-dims {act_pos_dim}+{act_rot_dim}+{act_grip_dim} "
                f"must sum to action_dim={action_dim}")
        if self.has_proprio:
            assert prop_pos_dim + prop_rot_dim + prop_grip_dim == gripper_dim, (
                f"proprio sub-dims {prop_pos_dim}+{prop_rot_dim}+{prop_grip_dim} "
                f"must sum to gripper_dim={gripper_dim}")

        # Generalized per-group token splitting. When *_token_groups are given
        # (lists of sub-dims summing to action_dim / gripper_dim), each group is
        # its own token with its own projection, type-encoding and decoder head.
        # Generalizes split_action_tokens: [3,6,1] reproduces pos/rot/grip,
        # [1,1,...,1] gives one token per scalar dimension. Opt-in: both None ->
        # legacy named-module path runs unchanged (preserves checkpoints).
        self.grouped_tokens = (action_token_groups is not None) or (proprio_token_groups is not None)
        if self.grouped_tokens:
            # 'per_dim' is a dim-agnostic sentinel: one token per scalar dimension.
            # The same config value works across policies with different action_dim
            # (e.g. MimicGen rot3 vs RLBench rot6d) -> truly uniform tokenization.
            if action_token_groups == 'per_dim':
                action_token_groups = [1] * action_dim
            elif action_token_groups is None:
                action_token_groups = ([act_pos_dim, act_rot_dim, act_grip_dim]
                                       if self.split_action_tokens else [action_dim])
            self.action_token_groups = [int(d) for d in action_token_groups]
            assert sum(self.action_token_groups) == action_dim, (
                f"action_token_groups {self.action_token_groups} must sum to action_dim={action_dim}")
            if self.has_proprio:
                if proprio_token_groups == 'per_dim':
                    proprio_token_groups = [1] * gripper_dim
                elif proprio_token_groups is None:
                    proprio_token_groups = [prop_pos_dim, prop_rot_dim, prop_grip_dim]
                self.proprio_token_groups = [int(d) for d in proprio_token_groups]
                assert sum(self.proprio_token_groups) == gripper_dim, (
                    f"proprio_token_groups {self.proprio_token_groups} must sum to gripper_dim={gripper_dim}")
            else:
                self.proprio_token_groups = []

        # Auxiliary action tokenizations. Opt-in; `None` leaves the module's
        # state_dict byte-identical to a run without this feature.
        self.aux_action_token_groups = _normalize_aux_action_groups(
            aux_action_token_groups, action_dim)
        # Which head's decode goes into x_out (i.e. what actually gets executed
        # at rollout). None = the primary branch; an int selects that auxiliary
        # branch instead. Runtime-only, adds no parameters, so it can be flipped
        # on an already-trained checkpoint to ask "what if we ran the other
        # tokenization's head?" without retraining.
        self.execute_aux_branch = None
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

        # Task-ID embedding (multitask). When n_tasks <= 1 the lookup is a no-op
        # zero vector and the per-token output is unchanged.
        self.n_tasks = int(n_tasks)
        if self.n_tasks > 1:
            self.task_embedding = nn.Embedding(self.n_tasks, projection_dim)
            nn.init.normal_(self.task_embedding.weight, std=0.02)
        else:
            self.task_embedding = None

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

        def _make_group_encodings(n):
            return nn.ParameterList(
                [nn.Parameter(0.02 * torch.randn(1, 1, projection_dim)) for _ in range(n)])

        if self.grouped_tokens:
            self.a_group_projections = nn.ModuleList([_make_proj(d) for d in self.action_token_groups])
            self.a_group_encodings = _make_group_encodings(len(self.action_token_groups))
            if self.has_proprio:
                self.p_group_projections = nn.ModuleList([_make_proj(d) for d in self.proprio_token_groups])
                self.p_group_encodings = _make_group_encodings(len(self.proprio_token_groups))
        elif self.split_action_tokens:
            self.a_pos_projection = _make_proj(act_pos_dim)
            self.a_rot_projection = _make_proj(act_rot_dim)
            self.a_grip_projection = _make_proj(act_grip_dim)
            self.a_pos_encoding = nn.Parameter(0.02 * torch.randn(1, 1, projection_dim))
            self.a_rot_encoding = nn.Parameter(0.02 * torch.randn(1, 1, projection_dim))
            self.a_grip_encoding = nn.Parameter(0.02 * torch.randn(1, 1, projection_dim))
        else:
            self.action_projection = _make_proj(action_dim)
            self.action_encoding = nn.Parameter(0.02 * torch.randn(1, 1, projection_dim))

        if self.has_proprio and not self.grouped_tokens:
            self.p_pos_projection = _make_proj(prop_pos_dim)
            self.p_rot_projection = _make_proj(prop_rot_dim)
            self.p_grip_projection = _make_proj(prop_grip_dim)
            self.p_pos_encoding = nn.Parameter(0.02 * torch.randn(1, 1, projection_dim))
            self.p_rot_encoding = nn.Parameter(0.02 * torch.randn(1, 1, projection_dim))
            self.p_grip_encoding = nn.Parameter(0.02 * torch.randn(1, 1, projection_dim))

        if self.aux_action_token_groups:
            self.aux_a_projections = nn.ModuleList(
                [nn.ModuleList([_make_proj(d) for d in groups])
                 for groups in self.aux_action_token_groups])
            self.aux_a_encodings = nn.ModuleList(
                [_make_group_encodings(len(groups))
                 for groups in self.aux_action_token_groups])

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

        if self.grouped_tokens:
            self.a_group_decoders = nn.ModuleList([_make_dec(d) for d in self.action_token_groups])
            if self.has_proprio:
                self.p_group_decoders = nn.ModuleList([_make_dec(d) for d in self.proprio_token_groups])
        elif self.split_action_tokens:
            self.a_pos_decoder = _make_dec(act_pos_dim)
            self.a_rot_decoder = _make_dec(act_rot_dim)
            self.a_grip_decoder = _make_dec(act_grip_dim)
        else:
            self.action_decoder = _make_dec(action_dim)

        if self.has_proprio and not self.grouped_tokens:
            self.p_pos_decoder = _make_dec(prop_pos_dim)
            self.p_rot_decoder = _make_dec(prop_rot_dim)
            self.p_grip_decoder = _make_dec(prop_grip_dim)

        if self.aux_action_token_groups:
            self.aux_a_decoders = nn.ModuleList(
                [nn.ModuleList([_make_dec(d) for d in groups])
                 for groups in self.aux_action_token_groups])

        # Particle encoding: either shared or view-specific for multi-view inputs.
        if self.multiview:
            self.view1_encoding = nn.Parameter(0.02 * torch.randn(1, 1, 1, projection_dim))
            self.view2_encoding = nn.Parameter(0.02 * torch.randn(1, 1, 1, projection_dim))
        else:
            self.particle_encoding = nn.Parameter(0.02 * torch.randn(1, 1, 1, projection_dim))

    def forward(self, x, cond, time, task_id=None, return_attention=False,
                return_aux=False):
        """
        Input/output flat layout (both paths):
            [action(action_dim), gripper(gripper_dim), bg(bg_dim), particles(K*features_dim)]

        When gripper_dim > 0, action and gripper are each split into three
        sub-components (pos, rot, grip) and enter the transformer as six
        separate tokens followed by the K particle tokens. The output is
        reassembled into the original flat layout.

        `return_aux=True` additionally returns the auxiliary action branches'
        decodes as a list of [bs, T, action_dim] tensors (training only —
        `x_out` is always the primary branch's decode).
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

        # Add task embedding to the diffusion-time embedding before AdaLN
        # gating. Both are summed into every token below.
        if self.task_embedding is not None and task_id is not None:
            task_id_long = task_id.to(dtype=torch.long, device=t_embed.device).view(-1)
            task_embed = self.task_embedding(task_id_long)  # [bs, projection_dim]
            t_embed = t_embed + task_embed

        # Build robot tokens piecewise: action side first, then proprio side.
        robot_tokens = []
        if self.grouped_tokens:
            action_slice = x[:, :, :self.action_dim]
            anchor_pool = []  # leading position dims form the spatial AdaLN anchor
            off = 0
            for d, proj, enc in zip(self.action_token_groups,
                                    self.a_group_projections, self.a_group_encodings):
                tok = proj(action_slice[:, :, off:off + d]) + enc.repeat(bs, T, 1)
                robot_tokens.append(tok)
                if off + d <= self.act_pos_dim:
                    anchor_pool.append(tok)
                off += d
            anchor = torch.stack(anchor_pool, dim=0).mean(0) if anchor_pool else robot_tokens[0]
        elif self.split_action_tokens:
            ap0 = 0
            ap1 = ap0 + self.act_pos_dim
            ar1 = ap1 + self.act_rot_dim
            ag1 = ar1 + self.act_grip_dim  # == action_dim
            a_pos = x[:, :, ap0:ap1]
            a_rot = x[:, :, ap1:ar1]
            a_grip = x[:, :, ar1:ag1]
            a_pos_tok = self.a_pos_projection(a_pos) + self.a_pos_encoding.repeat(bs, T, 1)
            a_rot_tok = self.a_rot_projection(a_rot) + self.a_rot_encoding.repeat(bs, T, 1)
            a_grip_tok = self.a_grip_projection(a_grip) + self.a_grip_encoding.repeat(bs, T, 1)
            robot_tokens.extend([a_pos_tok, a_rot_tok, a_grip_tok])
            anchor = a_pos_tok
        else:
            actions = x[:, :, :self.action_dim]
            action_tok = self.action_projection(actions) + self.action_encoding.repeat(bs, T, 1)
            robot_tokens.append(action_tok)
            anchor = action_tok

        if self.has_proprio:
            if self.grouped_tokens:
                off = self.action_dim
                for d, proj, enc in zip(self.proprio_token_groups,
                                        self.p_group_projections, self.p_group_encodings):
                    tok = proj(x[:, :, off:off + d]) + enc.repeat(bs, T, 1)
                    robot_tokens.append(tok)
                    off += d
            else:
                pp0 = self.action_dim
                pp1 = pp0 + self.prop_pos_dim
                pr1 = pp1 + self.prop_rot_dim
                pg1 = pr1 + self.prop_grip_dim  # == action_dim + gripper_dim
                p_pos = x[:, :, pp0:pp1]
                p_rot = x[:, :, pp1:pr1]
                p_grip = x[:, :, pr1:pg1]
                p_pos_tok = self.p_pos_projection(p_pos) + self.p_pos_encoding.repeat(bs, T, 1)
                p_rot_tok = self.p_rot_projection(p_rot) + self.p_rot_encoding.repeat(bs, T, 1)
                p_grip_tok = self.p_grip_projection(p_grip) + self.p_grip_encoding.repeat(bs, T, 1)
                robot_tokens.extend([p_pos_tok, p_rot_tok, p_grip_tok])

        # Auxiliary action branches: re-tokenize the SAME action slice. Appended
        # last so the primary token indices match a run without them. Always
        # built (they are part of the attended token set at sampling time too);
        # only their decoders are conditional on `return_aux`.
        n_primary_tokens = len(robot_tokens)
        if self.aux_action_token_groups:
            action_slice = x[:, :, :self.action_dim]
            for groups, projs, encs in zip(self.aux_action_token_groups,
                                           self.aux_a_projections,
                                           self.aux_a_encodings):
                off = 0
                for d, proj, enc in zip(groups, projs, encs):
                    robot_tokens.append(
                        proj(action_slice[:, :, off:off + d]) + enc.repeat(bs, T, 1))
                    off += d

        x_cat = torch.cat(
            [tok.unsqueeze(2) for tok in robot_tokens] + [new_state_particles],
            dim=2,
        )
        particle_start_token_idx = len(robot_tokens)

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

        # decode the auxiliary branches first when one of them is to be executed
        _aux_decoded = []
        if (self.execute_aux_branch is not None) and self.aux_action_token_groups:
            _ai = n_primary_tokens
            for _decs in self.aux_a_decoders:
                _bp = []
                for _dec in _decs:
                    _bp.append(_dec(particles_trans[:, :, _ai, :])); _ai += 1
                _aux_decoded.append(torch.cat(_bp, dim=-1))

        parts = []
        idx = 0
        if self.grouped_tokens:
            for dec in self.a_group_decoders:
                parts.append(dec(particles_trans[:, :, idx, :])); idx += 1
        elif self.split_action_tokens:
            parts.append(self.a_pos_decoder(particles_trans[:, :, idx, :])); idx += 1
            parts.append(self.a_rot_decoder(particles_trans[:, :, idx, :])); idx += 1
            parts.append(self.a_grip_decoder(particles_trans[:, :, idx, :])); idx += 1
        else:
            parts.append(self.action_decoder(particles_trans[:, :, idx, :])); idx += 1

        # swap the executed action for the chosen auxiliary head's decode
        if _aux_decoded:
            k = int(self.execute_aux_branch)
            assert 0 <= k < len(_aux_decoded), (
                f"execute_aux_branch={k} out of range (have {len(_aux_decoded)} aux branches)")
            parts = [_aux_decoded[k]]

        if self.has_proprio:
            if self.grouped_tokens:
                for dec in self.p_group_decoders:
                    parts.append(dec(particles_trans[:, :, idx, :])); idx += 1
            else:
                parts.append(self.p_pos_decoder(particles_trans[:, :, idx, :])); idx += 1
                parts.append(self.p_rot_decoder(particles_trans[:, :, idx, :])); idx += 1
                parts.append(self.p_grip_decoder(particles_trans[:, :, idx, :])); idx += 1

        if self.bg_dim > 0 and bg_features is not None:
            parts.append(bg_features)
        parts.append(particle_decoder_out)
        x_out = torch.cat(parts, dim=-1)

        aux_preds = []
        if return_aux and self.aux_action_token_groups:
            if _aux_decoded:
                aux_preds = _aux_decoded
            else:
                aux_idx = n_primary_tokens
                for decs in self.aux_a_decoders:
                    branch_parts = []
                    for dec in decs:
                        branch_parts.append(dec(particles_trans[:, :, aux_idx, :]))
                        aux_idx += 1
                    aux_preds.append(torch.cat(branch_parts, dim=-1))

        if return_attention and return_aux:
            return x_out, attention_dict, aux_preds
        if return_attention:
            return x_out, attention_dict
        if return_aux:
            return x_out, aux_preds
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
