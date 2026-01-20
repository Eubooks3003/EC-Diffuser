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

class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_path='', dataset_name='panda_push', horizon=64, obs_only=False,
        normalizer='LimitsNormalizer', particle_normalizer='ParticleGaussianNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=5000, termination_penalty=0, use_padding=True, overfit=False, action_only=False, single_view=False,
        action_z_scale=1.0, use_gripper_obs=False, **kwargs):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, dataset_name)
        self.dataset_path = dataset_path
        self.horizon = horizon
        self.obs_only = obs_only
        self.action_only = action_only
        self.max_path_length = max_path_length
        self.use_padding = use_padding
        self.action_z_scale = float(action_z_scale)  # Scale factor for Z action dimension
        self.use_gripper_obs = use_gripper_obs  # Whether to include gripper state in observations

        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        assert dataset_path, 'Dataset path must be provided'
        fields.load_paths_from_pickle(dataset_path, single_view=single_view and 'kitchen' not in dataset_path)
        if overfit:
            fields._count = 1
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
            normalize_keys = ['observations', 'actions', 'goals']
        if self.use_gripper_obs and self.has_gripper_state:
            normalize_keys.append('gripper_state')
        self.normalize(keys=normalize_keys)

        self.particle_dim = fields.observations.shape[-1]
        self.observation_dim = fields.normed_observations.shape[-1]
        self.action_dim = fields.normed_actions.shape[-1]
        self.gripper_dim = fields.gripper_state.shape[-1] if (self.use_gripper_obs and self.has_gripper_state) else 0
        print(f'[ datasets/sequence ] Dataset fields: {self.fields}')
        print(f'[ datasets/sequence ] Dataset normalizer: {self.normalizer}')
        print(f'[ datasets/sequence ] action_dim={self.action_dim}, gripper_dim={self.gripper_dim}, observation_dim={self.observation_dim}')

    def normalize(self, keys=['observations', 'actions', 'goals']):
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

    def get_conditions(self, observations, gripper_state=None):
        '''
            condition on current observation for planning
            If gripper_state is provided, it's concatenated with observations in conditions
        '''
        if gripper_state is not None:
            # Conditions include gripper state: [gripper_state, observations]
            return {0: np.concatenate([gripper_state[0], observations[0]], axis=-1)}
        return {0: observations[0]}

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

        conditions = self.get_conditions(observations, gripper_state)
        if self.obs_only:
            trajectories = observations
        else:
            # Trajectory format: [actions, gripper_state (optional), observations]
            if gripper_state is not None:
                trajectories = np.concatenate([actions, gripper_state, observations], axis=-1)
            else:
                trajectories = np.concatenate([actions, observations], axis=-1)
        batch = Batch(trajectories, conditions)
        return batch


class GoalDataset(SequenceDataset):

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = path_length - 1
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices


    def __getitem__(self, idx):
        path_ind, start, end = self.indices[idx]
        path_length = self.fields.path_lengths[path_ind]
        # hindsight goals for unsuccessful episodes
        if path_ind in self.successful_episode_idxes:
            goal = self.fields.normed_goals[path_ind, path_length-1:path_length]
        else:
            goal = self.fields.normed_observations[path_ind, path_length-1:path_length]

        # Calculate actual end and determine if padding is needed
        path_length = self.fields.path_lengths[path_ind]
        actual_end = min(end, path_length)
        padding_needed = end - actual_end

        # Fetch observations and actions
        if self.action_only:
            observations = np.concatenate([self.fields.normed_observations[path_ind, start:start+1].repeat(actual_end-1-start, axis=0),
                                           goal])
        else:
            observations = np.concatenate([self.fields.normed_observations[path_ind, start:actual_end-1],
                                           goal])
        actions = np.concatenate([self.fields.normed_actions[path_ind, start:actual_end-1],
                                  self.normalizer.normalize(np.zeros((1, self.action_dim), dtype=self.fields.normed_actions.dtype), 'actions')])

        # Get gripper state if available and requested
        if self.use_gripper_obs and self.has_gripper_state:
            # Get gripper state for the goal (last timestep)
            goal_gripper = self.fields.normed_gripper_state[path_ind, path_length-1:path_length]
            gripper_state = np.concatenate([
                self.fields.normed_gripper_state[path_ind, start:actual_end-1],
                goal_gripper
            ])
        else:
            gripper_state = None

        # Handle padding
        if padding_needed > 0:
            # Pad observations with the last observation repeated
            last_obs = observations[-1]
            observations = np.vstack([observations] + [last_obs] * padding_needed)
            # Pad actions with zeros
            last_action = actions[-1]
            actions = np.vstack([actions] + [last_action] * padding_needed)
            # Pad gripper state if present
            if gripper_state is not None:
                last_gripper = gripper_state[-1]
                gripper_state = np.vstack([gripper_state] + [last_gripper] * padding_needed)

        conditions = self.get_conditions(observations, gripper_state)
        if self.obs_only:
            trajectories = observations
        else:
            # Trajectory format: [actions, gripper_state (optional), observations]
            if gripper_state is not None:
                trajectories = np.concatenate([actions, gripper_state, observations], axis=-1)
            else:
                trajectories = np.concatenate([actions, observations], axis=-1)
        batch = Batch(trajectories, conditions)
        return batch

    def get_conditions(self, observations, gripper_state=None):
        '''
            condition on both the current observation and the last observation in the plan
            If gripper_state is provided, it's concatenated with observations in conditions
        '''
        if gripper_state is not None:
            return {
                0: np.concatenate([gripper_state[0], observations[0]], axis=-1).copy(),
                self.horizon - 1: np.concatenate([gripper_state[-1], observations[-1]], axis=-1).copy(),
            }
        return {
            0: observations[0].copy(),
            self.horizon - 1: observations[-1].copy(),
        }