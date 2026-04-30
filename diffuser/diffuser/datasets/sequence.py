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


Batch = namedtuple('Batch', 'trajectories conditions')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')
MultitaskBatch = namedtuple('MultitaskBatch', 'trajectories conditions task_id')

class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_path='', dataset_name='panda_push', horizon=64, obs_only=False,
        normalizer='LimitsNormalizer', particle_normalizer='ParticleGaussianNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=5000, termination_penalty=0, use_padding=True, overfit=False, action_only=False, single_view=False,
        action_z_scale=1.0, use_gripper_obs=False, use_bg_obs=False, **kwargs):
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
        self.normalizer = DatasetNormalizer(fields, normalizer, particle_normalizer=particle_normalizer, path_lengths=fields['path_lengths'])

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
        batch = Batch(trajectories, conditions)
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
        batch = Batch(trajectories, conditions)
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


class MultitaskGoalDataset(GoalDataset):
    """
    GoalDataset extension for multitask training.

    Loads N per-task pkls into a single buffer via
    `ReplayBuffer.load_paths_from_pickles`, fits one global normalizer on
    valid frames across all tasks, and returns `MultitaskBatch(trajectories,
    conditions, task_id)` so the diffusion model can be conditioned on task.
    """

    def __init__(self, dataset_path='', dataset_name='multitask', horizon=64, obs_only=False,
                 normalizer='LimitsNormalizer', particle_normalizer='ParticleGaussianNormalizer',
                 preprocess_fns=[], max_path_length=1000, max_n_episodes=5000,
                 termination_penalty=0, use_padding=True, overfit=False, action_only=False,
                 single_view=False, action_z_scale=1.0, use_gripper_obs=False,
                 use_bg_obs=False, task_entries=None, max_demos_per_task=None, **kwargs):
        if task_entries is None:
            raise ValueError("MultitaskGoalDataset requires `task_entries` (list of {name, task_id, pkl, ...}).")

        self.preprocess_fn = get_preprocess_fn(preprocess_fns, dataset_name)
        self.dataset_path = '|'.join([e['pkl'] for e in task_entries])
        self.horizon = horizon
        self.obs_only = obs_only
        self.action_only = action_only
        self.max_path_length = max_path_length
        self.use_padding = use_padding
        self.action_z_scale = float(action_z_scale)
        self.use_gripper_obs = use_gripper_obs
        self.use_bg_obs = use_bg_obs
        self.task_entries = task_entries
        self.task_names = [e['name'] for e in task_entries]
        self.task_name_to_id = {e['name']: int(e['task_id']) for e in task_entries}

        # Load all pkls into one buffer with a `task_ids` field.
        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        pkls_with_ids = [(e['pkl'], int(e['task_id'])) for e in task_entries]
        fields.load_paths_from_pickles(
            pkls_with_ids,
            max_demos_per_task=max_demos_per_task,
            single_view=single_view,
        )
        if overfit:
            fields._count = 1
        fields.finalize()

        # Buffer T-dim is the global max across loaded tasks. Override so the
        # inherited normalize/reshape uses the correct allocated length.
        actual_T = fields['observations'].shape[1]
        if actual_T != self.max_path_length:
            print(f'[ datasets/sequence ] Multitask: overriding max_path_length '
                  f'{self.max_path_length} -> {actual_T} (global max across tasks)')
            self.max_path_length = actual_T

        # Mirror the single-task post-load pipeline.
        if self.action_z_scale != 1.0:
            print(f'[ datasets/sequence ] Applying Z action scaling: {self.action_z_scale}x')
            fields['actions'][:, :, 2] *= self.action_z_scale

        self.has_gripper_state = 'gripper_state' in fields._dict
        if self.use_gripper_obs and not self.has_gripper_state:
            print(f'[ datasets/sequence ] WARNING: use_gripper_obs=True but no gripper_state in dataset')
            self.use_gripper_obs = False

        self.has_bg_features = 'bg_features' in fields._dict
        if self.use_bg_obs and not self.has_bg_features:
            print(f'[ datasets/sequence ] WARNING: use_bg_obs=True but no bg_features in dataset')
            self.use_bg_obs = False

        self.successful_episode_idxes = getattr(fields, 'successful_episode_idxes', np.array([]))

        # Single global normalizer fit on valid frames across all tasks.
        # `flatten()` already respects path_lengths so padded zeros are excluded.
        self.normalizer = DatasetNormalizer(
            fields, normalizer, particle_normalizer=particle_normalizer,
            path_lengths=fields['path_lengths'],
        )

        self.normalizer.sanity_check_roundtrip(
            {'actions': fields['actions'], 'observations': fields['observations']},
            n_samples=3, save_path=None,
        )

        self.indices = self.make_indices(fields.path_lengths, horizon)
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths

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
        print(f'[ datasets/sequence ] Multitask dataset fields: {self.fields}')
        print(f'[ datasets/sequence ] Multitask normalizer: {self.normalizer}')
        print(f'[ datasets/sequence ] action_dim={self.action_dim}, gripper_dim={self.gripper_dim}, '
              f'bg_dim={self.bg_dim}, observation_dim={self.observation_dim}, '
              f'n_tasks={len(self.task_entries)}')

    def __getitem__(self, idx):
        # Reuse the GoalDataset slicing/padding logic, then attach task_id.
        path_ind, start, end = self.indices[idx]
        path_length = self.fields.path_lengths[path_ind]

        actual_end = min(end, path_length)
        padding_needed = end - actual_end

        observations = self.fields.normed_observations[path_ind, start:actual_end]
        actions = self.fields.normed_actions[path_ind, start:actual_end]

        if self.use_gripper_obs and self.has_gripper_state:
            gripper_state = self.fields.normed_gripper_state[path_ind, start:actual_end]
        else:
            gripper_state = None

        if self.use_bg_obs and self.has_bg_features:
            bg_features = self.fields.normed_bg_features[path_ind, start:actual_end]
        else:
            bg_features = None

        if padding_needed > 0:
            observations = np.vstack([observations] + [observations[-1:]] * padding_needed)
            actions = np.vstack([actions] + [actions[-1:]] * padding_needed)
            if gripper_state is not None:
                gripper_state = np.vstack([gripper_state] + [gripper_state[-1:]] * padding_needed)
            if bg_features is not None:
                bg_features = np.vstack([bg_features] + [bg_features[-1:]] * padding_needed)

        conditions = self.get_conditions(observations, gripper_state, bg_features)
        if self.obs_only:
            trajectories = observations
        else:
            traj_parts = [actions]
            if gripper_state is not None:
                traj_parts.append(gripper_state)
            if bg_features is not None:
                traj_parts.append(bg_features)
            traj_parts.append(observations)
            trajectories = np.concatenate(traj_parts, axis=-1)

        task_id = np.int64(self.fields.task_ids[path_ind])
        return MultitaskBatch(trajectories, conditions, task_id)