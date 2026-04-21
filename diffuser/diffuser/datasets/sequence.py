from collections import namedtuple
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import torch
import pdb

from .preprocessing import get_preprocess_fn
# from .d4rl import load_environment, sequence_dataset

# try:
#     from .d4rl import load_environment, sequence_dataset
# except Exception as e:
#     load_environment = None
#     sequence_dataset = None
#     _D4RL_IMPORT_ERROR = e

from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer


Batch = namedtuple('Batch', ['trajectories', 'conditions', 'action_conditions', 'lang', 'lang_mask'])
Batch.__new__.__defaults__ = (None, None, None)  # action_conditions / lang / lang_mask default to None
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')

class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_path='', dataset_name='panda_push', horizon=64, obs_only=False,
        normalizer='LimitsNormalizer', particle_normalizer='ParticleGaussianNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=5000, termination_penalty=0, use_padding=True, overfit=False, action_only=False, single_view=False,
        action_z_scale=1.0, use_gripper_obs=False, use_bg_obs=False,
        use_views=None, num_source_views=None,
        action_normalizer=None, **kwargs):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, dataset_name)
        self.dataset_path = dataset_path
        self.horizon = horizon
        self.obs_only = obs_only
        self.action_only = action_only
        self.max_path_length = max_path_length
        self.use_padding = use_padding
        self.action_z_scale = float(action_z_scale)  # Scale factor for Z action dimension
        self.use_gripper_obs = use_gripper_obs  # Whether to include gripper state in observations
        self.use_bg_obs = use_bg_obs  # Whether to include background features in observations

        max_demos = kwargs.pop('max_demos', None)
        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        assert dataset_path, 'Dataset path must be provided'
        fields.load_paths_from_pickle(dataset_path, single_view=single_view and 'kitchen' not in dataset_path)
        if overfit:
            fields._count = 1
        if max_demos is not None and max_demos < fields._count:
            print(f'[ datasets/sequence ] Limiting to {max_demos}/{fields._count} demos')
            fields._count = max_demos
        fields.finalize()

        # Optional view subsetting: the pkl stores tokens stacked in view order
        # (view0 particles | view1 particles | ...). If the caller wants to train
        # on a subset of views (e.g. only 'front' from a 4-view pkl), slice the
        # observations and bg_features in-place here.
        if use_views is not None and 'observations' in fields._dict:
            obs = fields._dict['observations']       # (E, T, K_total, Dtok)
            K_total = obs.shape[2]
            Nv = int(num_source_views) if num_source_views else None
            if Nv is None or K_total % Nv != 0:
                # Infer from K_total if not provided and Nv divides evenly
                raise ValueError(
                    f"use_views requires num_source_views (pkl total views) to be set. "
                    f"K_total={K_total}, got num_source_views={num_source_views}"
                )
            K_per_view = K_total // Nv
            view_idx = list(use_views)
            particle_indices = np.concatenate([
                np.arange(v * K_per_view, (v + 1) * K_per_view) for v in view_idx
            ])
            fields._dict['observations'] = obs[:, :, particle_indices, :].copy()
            print(f'[ datasets/sequence ] use_views={view_idx}: sliced observations '
                  f'{obs.shape} -> {fields._dict["observations"].shape}')
            # Slice bg_features similarly (stored as (E, T, Nv*bg_per_view))
            if 'bg_features' in fields._dict:
                bg = fields._dict['bg_features']
                bg_total = bg.shape[-1]
                if bg_total % Nv == 0:
                    bg_per_view = bg_total // Nv
                    bg_indices = np.concatenate([
                        np.arange(v * bg_per_view, (v + 1) * bg_per_view) for v in view_idx
                    ])
                    fields._dict['bg_features'] = bg[:, :, bg_indices].copy()
                    print(f'[ datasets/sequence ] use_views={view_idx}: sliced bg_features '
                          f'{bg.shape} -> {fields._dict["bg_features"].shape}')

        # Apply Z scaling to actions before normalization
        if self.action_z_scale != 1.0:
            print(f'[ datasets/sequence ] Applying Z action scaling: {self.action_z_scale}x')
            print(f'  Before scaling - Z range: [{fields["actions"][:,:,2].min():.4f}, {fields["actions"][:,:,2].max():.4f}]')
            fields['actions'][:, :, 2] *= self.action_z_scale
            print(f'  After scaling  - Z range: [{fields["actions"][:,:,2].min():.4f}, {fields["actions"][:,:,2].max():.4f}]')

        # Check for gripper state
        self.has_gripper_state = 'gripper_state' in fields._dict
        if self.use_gripper_obs and not self.has_gripper_state:
            print(f'[ datasets/sequence ] WARNING: use_gripper_obs=True but no gripper_state in dataset')
            self.use_gripper_obs = False
        if self.has_gripper_state:
            print(f'[ datasets/sequence ] Found gripper_state in dataset: shape={fields["gripper_state"].shape}')
            if self.use_gripper_obs:
                print(f'[ datasets/sequence ] Gripper state will be included in model input')
            else:
                print(f'[ datasets/sequence ] Gripper state available but not used (use_gripper_obs=False)')

        # Check for bg_features
        self.has_bg_features = 'bg_features' in fields._dict
        if self.use_bg_obs and not self.has_bg_features:
            print(f'[ datasets/sequence ] WARNING: use_bg_obs=True but no bg_features in dataset')
            self.use_bg_obs = False
        if self.has_bg_features:
            print(f'[ datasets/sequence ] Found bg_features in dataset: shape={fields["bg_features"].shape}')
            if self.use_bg_obs:
                print(f'[ datasets/sequence ] Background features will be included in model input')
            else:
                print(f'[ datasets/sequence ] Background features available but not used (use_bg_obs=False)')

        self.successful_episode_idxes = fields.successful_episode_idxes

        # OCGS-style action convention: action[t] = current pose, not next pose.
        # The pkl was preprocessed with action[t] = gripper_state[t+1] (next pose,
        # see lpwm-copy/scripts/ec_diffuser_rlbench_multiview_preprocess.py:171-179),
        # but the in-trainer eval at utils/training.py:817-828 pins action[0] to
        # the *current* gripper pose from the env and executes traj[1]. Training
        # and eval therefore see different distributions for the action[0] pin.
        # Rebind actions to gripper_state so action[t] = current pose at frame t
        # (matching OCGS scripts/preprocess_data.py:255). Must happen before the
        # normalizer below so action statistics are computed from the new labels.
        if self.has_gripper_state:
            fields._dict['actions'] = fields._dict['gripper_state'].copy()
            print('[ datasets/sequence ] Rebinding actions <- gripper_state '
                  '(current-pose convention; matches eval pin)')

        self.normalizer = DatasetNormalizer(fields, normalizer, particle_normalizer=particle_normalizer, path_lengths=fields['path_lengths'], action_normalizer=action_normalizer)

        # Sanity check: verify normalize -> unnormalize round-trip
        self.normalizer.sanity_check_roundtrip(
            {'actions': fields['actions'], 'observations': fields['observations']},
            n_samples=3,
            save_path=None  # Set to a path to save detailed report
        )

        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths

        # Determine which keys to normalize
        if 'kitchen' in dataset_path:
            normalize_keys = ['observations', 'actions']
        else:
            normalize_keys = ['observations', 'actions']
        if self.use_gripper_obs and self.has_gripper_state:
            normalize_keys.append('gripper_state')
        if self.use_bg_obs and self.has_bg_features:
            normalize_keys.append('bg_features')
        self.normalize(keys=normalize_keys)

        self.particle_dim = fields.observations.shape[-1]
        self.observation_dim = fields.normed_observations.shape[-1]
        self.action_dim = fields.normed_actions.shape[-1]
        self.gripper_dim = fields.gripper_state.shape[-1] if (self.use_gripper_obs and self.has_gripper_state) else 0
        self.bg_dim = fields.bg_features.shape[-1] if (self.use_bg_obs and self.has_bg_features) else 0
        print(f'[ datasets/sequence ] Dataset fields: {self.fields}')
        print(f'[ datasets/sequence ] Dataset normalizer: {self.normalizer}')
        print(f'[ datasets/sequence ] action_dim={self.action_dim}, gripper_dim={self.gripper_dim}, bg_dim={self.bg_dim}, observation_dim={self.observation_dim}')

    def normalize(self, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            if key not in self.fields._dict:
                continue
            if key == 'observations' or key == 'goals':
                array = self.fields[key]    # (n_episodes, max_path_length, n_entities, dim)
                normed = self.normalizer(array, 'observations')
            elif key == 'gripper_state':
                # Gripper state: (n_episodes, max_path_length, gripper_dim)
                array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
                normed = self.normalizer(array, key)
            else:
                array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
                normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)
    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations, gripper_state=None, bg_features=None):
        '''
            condition on current observation for planning
            Conditions are concatenated: [gripper_state (optional), bg_features (optional), observations]
        '''
        parts = []
        if gripper_state is not None:
            parts.append(gripper_state[0])
        if bg_features is not None:
            parts.append(bg_features[0])
        parts.append(observations[0])
        return {0: np.concatenate(parts, axis=-1)}

    def get_action_conditions(self, actions):
        '''Proprioception conditioning: pin action at t=0 to demo's action[0].'''
        return {0: actions[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path_ind, start, end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]

        # Get gripper state if available and requested
        if self.use_gripper_obs and self.has_gripper_state:
            gripper_state = self.fields.normed_gripper_state[path_ind, start:end]
        else:
            gripper_state = None

        # Get bg_features if available and requested
        if self.use_bg_obs and self.has_bg_features:
            bg_features = self.fields.normed_bg_features[path_ind, start:end]
        else:
            bg_features = None

        conditions = self.get_conditions(observations, gripper_state, bg_features)
        action_conditions = self.get_action_conditions(actions)
        if self.obs_only:
            trajectories = observations
        else:
            # Trajectory format: [actions, gripper_state (optional), bg_features (optional), observations]
            traj_parts = [actions]
            if gripper_state is not None:
                traj_parts.append(gripper_state)
            if bg_features is not None:
                traj_parts.append(bg_features)
            traj_parts.append(observations)
            trajectories = np.concatenate(traj_parts, axis=-1)
        batch = Batch(trajectories, conditions, action_conditions)
        return batch


class GoalDataset(SequenceDataset):

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint.
            max_start = path_length - horizon so every window is fully within bounds.
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = max(path_length - horizon, 0)
            for start in range(max_start + 1):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices


    def __getitem__(self, idx):
        path_ind, start, end = self.indices[idx]
        path_length = self.fields.path_lengths[path_ind]

        # Use actual consecutive frames — no goal substitution
        actual_end = min(end, path_length)
        padding_needed = end - actual_end

        # Fetch actual consecutive observations and actions
        observations = self.fields.normed_observations[path_ind, start:actual_end]
        actions = self.fields.normed_actions[path_ind, start:actual_end]

        # Get gripper state if available and requested
        if self.use_gripper_obs and self.has_gripper_state:
            gripper_state = self.fields.normed_gripper_state[path_ind, start:actual_end]
        else:
            gripper_state = None

        # Get bg_features if available and requested
        if self.use_bg_obs and self.has_bg_features:
            bg_features = self.fields.normed_bg_features[path_ind, start:actual_end]
        else:
            bg_features = None

        # Handle padding (repeat last frame for windows that extend past episode end)
        if padding_needed > 0:
            last_obs = observations[-1:]
            observations = np.vstack([observations] + [last_obs] * padding_needed)
            last_action = actions[-1:]
            actions = np.vstack([actions] + [last_action] * padding_needed)
            if gripper_state is not None:
                last_gripper = gripper_state[-1:]
                gripper_state = np.vstack([gripper_state] + [last_gripper] * padding_needed)
            if bg_features is not None:
                last_bg = bg_features[-1:]
                bg_features = np.vstack([bg_features] + [last_bg] * padding_needed)

        conditions = self.get_conditions(observations, gripper_state, bg_features)
        action_conditions = self.get_action_conditions(actions)
        if self.obs_only:
            trajectories = observations
        else:
            # Trajectory format: [actions, gripper_state (optional), bg_features (optional), observations]
            traj_parts = [actions]
            if gripper_state is not None:
                traj_parts.append(gripper_state)
            if bg_features is not None:
                traj_parts.append(bg_features)
            traj_parts.append(observations)
            trajectories = np.concatenate(traj_parts, axis=-1)
        batch = Batch(trajectories, conditions, action_conditions)
        return batch

    def get_conditions(self, observations, gripper_state=None, bg_features=None):
        '''
            condition on current observation only (goal conditioning disabled).
            Goals are zeroed out since MimicGen env doesn't provide goal observations at inference.
            Conditions are concatenated: [gripper_state (optional), bg_features (optional), observations]
        '''
        parts_0 = []
        if gripper_state is not None:
            parts_0.append(gripper_state[0])
        if bg_features is not None:
            parts_0.append(bg_features[0])
        parts_0.append(observations[0])

        # Only condition on t=0, no goal conditioning
        return {
            0: np.concatenate(parts_0, axis=-1).copy(),
        }


class LanguageConditionedDataset(GoalDataset):
    """
    Drop-in replacement for GoalDataset that also yields a per-sample CLIP-encoded
    language instruction. Picks one of the N paraphrases available per episode
    uniformly at random each __getitem__ call (free text-level augmentation).

    Assumes the loaded pickle has a 'language' field of type list[list[str]] (one
    inner list per episode, each with >=1 paraphrases). Falls back to zeros if
    the field is missing.

    New kwargs:
        clip_model_name (str):     HuggingFace model id (default ViT-B/32)
        lang_device (str):         device for CLIP encoding (default 'cpu')
        lang_pooled (bool):        if True, use single EOS-pooled token; else
                                   full CLIP last_hidden_state sequence (default False)
        max_lang_tokens (int):     truncate CLIP sequence to at most this many
                                   tokens (default 32, saves compute)
    """

    def __init__(self, *args,
                 clip_model_name: str = "openai/clip-vit-base-patch32",
                 lang_device: str = "cpu",
                 lang_pooled: bool = False,
                 max_lang_tokens: int = 32,
                 **kwargs):
        super().__init__(*args, **kwargs)

        from diffuser.models.lang import CLIPTextEncoder, LanguageInstructionCache

        self.lang_pooled = lang_pooled
        self.max_lang_tokens = int(max_lang_tokens)

        if 'language' not in self.fields._dict:
            print('[ datasets/sequence ] WARNING: LanguageConditionedDataset but no '
                  '"language" field in pkl -- falling back to empty strings')
            self._has_lang = False
            self.lang_dim = 0
            return

        self._has_lang = True
        self._encoder = CLIPTextEncoder(
            model_name=clip_model_name, device=lang_device, return_pooled=lang_pooled,
        )
        self.lang_dim = self._encoder.clip_dim
        self._cache = LanguageInstructionCache(self._encoder)

        # Pre-encode every unique paraphrase across all episodes.
        unique = set()
        for ep_lang in self.fields.language:
            for s in ep_lang:
                unique.add(s)
        print(f'[ datasets/sequence ] Pre-encoding {len(unique)} unique language strings with CLIP...')
        for s in unique:
            self._cache.get(s)

    def _get_lang_tokens(self, path_ind):
        """Return (tokens, mask). tokens: (L, clip_dim). mask: (L,) 1=valid, 0=pad."""
        if not self._has_lang:
            return (torch.zeros((1, 1), dtype=torch.float32),
                    torch.zeros((1,), dtype=torch.float32))
        paraphrases = self.fields.language[path_ind]
        s = paraphrases[np.random.randint(len(paraphrases))]
        emb = self._cache.get(s)
        if self.lang_pooled:
            return emb.unsqueeze(0).float(), torch.ones((1,), dtype=torch.float32)
        tokens, mask = emb
        return (tokens[:self.max_lang_tokens].float(),
                mask[:self.max_lang_tokens].float())

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        path_ind = int(self.indices[idx][0])
        lang, lang_mask = self._get_lang_tokens(path_ind)
        return Batch(trajectories=batch.trajectories, conditions=batch.conditions,
                     action_conditions=batch.action_conditions,
                     lang=lang, lang_mask=lang_mask)
