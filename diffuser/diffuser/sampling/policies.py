from collections import namedtuple
import torch
import einops
import pdb

import diffuser.utils as utils
from diffuser.datasets.preprocessing import get_policy_preprocess_fn
import numpy as np
from diffuser.models import sample_fn_return_attn, default_sample_fn


Trajectories = namedtuple('Trajectories', 'actions observations values')


class GoalConditionedPolicy:
    def __init__(self, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = diffusion_model.action_dim
        self.preprocess_fn = get_policy_preprocess_fn(preprocess_fns)
        self.sample_kwargs = sample_kwargs

    def __call__(self, conditions, batch_size=1, verbose=True, return_attention=False):
        conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
        x0 = conditions[0]
        K = getattr(self.diffusion_model.model, "max_particles", None)
        D = getattr(self.diffusion_model.model, "features_dim", None)

        if x0.ndim == 3:
            # (B,K,D)
            multi_input = True
        elif x0.ndim == 2 and (K is not None) and (D is not None) and x0.shape == (K, D):
            # (K,D) single env particles
            multi_input = False
        elif x0.ndim == 2:
            # (B, obs_dim) flattened batch
            multi_input = True
        else:
            # (obs_dim,) single
            multi_input = False

        conditions = self._format_conditions(conditions, batch_size, multi_input=multi_input)
        
        if return_attention:
            samples, att_dict = self.diffusion_model(conditions, verbose=verbose, sort_by_value=False, return_attention=return_attention, **self.sample_kwargs)
            att_dict = {k: utils.to_np(v) for k, v in att_dict.items()}
        else:
            samples = self.diffusion_model(conditions, verbose=verbose, sort_by_value=False, return_attention=return_attention, **self.sample_kwargs)
        trajectories = utils.to_np(samples.trajectories)


        normed_observations = trajectories[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')    

        ## extract action [ batch_size x horizon x transition_dim ]
        actions = trajectories[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(actions, 'actions')
        ## extract first action per env if conditions contains multiple envs
        if not multi_input:
            action = actions[0, 0]
        else:
            action = actions[:, 0]


        trajectories = Trajectories(actions, observations, samples.values)
        if return_attention:
            return action, trajectories, att_dict
        return action, trajectories

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size, multi_input=False):
        K = getattr(self.diffusion_model.model, "max_particles", None)
        D = getattr(self.diffusion_model.model, "features_dim", None)

        def to_BKD(x):
            # returns either (B,K,D) if possible, else returns x unchanged
            if (K is None) or (D is None):
                return x

            if x.ndim == 1:
                # (K*D,) -> (1,K,D) if matches
                if x.shape[0] == K * D:
                    return x.reshape(1, K, D)
                return x

            if x.ndim == 2:
                # (K,D) single env particles
                if x.shape == (K, D):
                    return x.reshape(1, K, D)
                # (B, K*D) flattened batch
                if x.shape[1] == K * D:
                    return x.reshape(x.shape[0], K, D)
                return x

            if x.ndim == 3:
                # already (B,K,D)
                return x

            return x

        def to_flat(x):
            # (B,K,D) -> (B, K*D)
            if (K is None) or (D is None):
                return x
            if x.ndim == 3 and x.shape[-2] == K and x.shape[-1] == D:
                return x.reshape(x.shape[0], K * D)
            return x

        # 1) make sure observations/goals are (B,K,D) for the particle normalizer
        conditions = {k: to_BKD(v) for k, v in conditions.items()}

        # 2) normalize in particle space
        conditions = utils.apply_dict(self.normalizer.normalize, conditions, "observations")

        # 3) flatten back because diffusion model was trained on flattened obs
        conditions = {k: to_flat(v) for k, v in conditions.items()}

        # 4) torch + repeat if single-env
        conditions = utils.to_torch(conditions, dtype=torch.float32)
        if not multi_input:
            conditions = utils.apply_dict(
                einops.repeat,
                conditions,
                "b d -> (repeat b) d", repeat=batch_size,
            )
        return conditions
