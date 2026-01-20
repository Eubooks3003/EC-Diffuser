import numpy as np
import pickle

def atleast_2d(x):
    while x.ndim < 2:
        x = np.expand_dims(x, axis=-1)
    return x

class ReplayBuffer:

    def __init__(self, max_n_episodes, max_path_length, termination_penalty):
        self._dict = {
            'path_lengths': np.zeros(max_n_episodes, dtype=int),
        }
        self._count = 0
        self.max_n_episodes = max_n_episodes
        self.max_path_length = max_path_length
        self.termination_penalty = termination_penalty

    def __repr__(self):
        return '[ datasets/buffer ] Fields:\n' + '\n'.join(
            f'    {key}: {val.shape}'
            for key, val in self.items()
        )

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, val):
        self._dict[key] = val
        self._add_attributes()

    @property
    def n_episodes(self):
        return self._count

    @property
    def n_steps(self):
        return sum(self['path_lengths'])

    def _add_keys(self, path):
        if hasattr(self, 'keys'):
            return
        self.keys = list(path.keys())

    def _add_attributes(self):
        '''
            can access fields with `buffer.observations`
            instead of `buffer['observations']`
        '''
        for key, val in self._dict.items():
            setattr(self, key, val)

    def items(self):
        return {k: v for k, v in self._dict.items()
                if k not in ('path_lengths', 'meta')}.items()


    def _allocate(self, key, array):
        assert key not in self._dict
        dim = array.shape[-1]
        shape = (self.max_n_episodes, self.max_path_length, dim)
        self._dict[key] = np.zeros(shape, dtype=np.float32)
        # print(f'[ utils/mujoco ] Allocated {key} with size {shape}')

    def add_path(self, path):
        path_length = len(path['observations'])
        assert path_length <= self.max_path_length

        ## if first path added, set keys based on contents
        self._add_keys(path)

        ## add tracked keys in path
        for key in self.keys:
            array = atleast_2d(path[key])
            if key not in self._dict: self._allocate(key, array)
            self._dict[key][self._count, :path_length] = array

        ## penalize early termination
        if path['terminals'].any() and self.termination_penalty is not None:
            assert not path['timeouts'].any(), 'Penalized a timeout episode for early termination'
            self._dict['rewards'][self._count, path_length - 1] += self.termination_penalty

        ## record path length
        self._dict['path_lengths'][self._count] = path_length

        ## increment path counter
        self._count += 1

    def truncate_path(self, path_ind, step):
        old = self._dict['path_lengths'][path_ind]
        new = min(step, old)
        self._dict['path_lengths'][path_ind] = new

    def finalize(self):
        ## remove extra slots
        for key in self.keys + ['path_lengths']:
            self._dict[key] = self._dict[key][-self._count:]
        self._add_attributes()
        print(f'[ datasets/buffer ] Finalized replay buffer | {self._count} episodes')
        if 'info_goals_reached' in self._dict:
            self.successful_episode_idxes = np.where(self._dict['info_goals_reached'] == 1)[0]
        else:
            self.successful_episode_idxes = np.array([])
        print(f'[ datasets/buffer ] Found {len(self.successful_episode_idxes)} successful episodes')

    def load_paths_from_pickle(self, path, single_view=False):
        paths_dict = pickle.load(open(path, 'rb'))

        # allow meta dicts etc
        meta = paths_dict.get('meta', None)
        if isinstance(meta, dict):
            K_expected = meta.get('K', None) or meta.get('num_entity', None)
        else:
            K_expected = None

        for key, val in paths_dict.items():
            # keep meta as-is
            if key == 'meta':
                self._dict[key] = val
                continue

            # Single-view handling ONLY if data is actually packed as 2K
            if single_view and (key == 'observations' or key == 'goals'):
                if isinstance(val, np.ndarray) and val.ndim == 4:
                    # val: (E,T,K,D)
                    E, T, K, D = val.shape

                    # If we know expected K, only slice when K == 2*K_expected
                    if K_expected is not None:
                        if K == 2 * K_expected:
                            val = val[:, :, :K_expected, :]
                        elif K == K_expected:
                            pass
                        else:
                            raise ValueError(
                                f"[buffer] unexpected particle dim K={K}, expected {K_expected} or {2*K_expected}"
                            )
                    else:
                        # If we DON'T know expected K, do NOT slice.
                        # (Your dataset is already single-view K=64.)
                        pass

            if key == 'path_lengths':
                self._dict[key] = val.astype(np.int32)
            else:
                self._dict[key] = val.astype(np.float32)

        self._count = self._dict['observations'].shape[0]

        if 'path_lengths' not in paths_dict:
            self._dict['path_lengths'] = np.array([len(obs) for obs in self._dict['observations']])

        self.keys = [k for k in paths_dict.keys() if k != 'meta']

        # Report gripper state if present
        if 'gripper_state' in self._dict:
            gs = self._dict['gripper_state']
            print(f'[ datasets/buffer ] Found gripper_state: shape={gs.shape}, '
                  f'range=[{gs.min():.3f}, {gs.max():.3f}]')

        # Report bg_features if present
        if 'bg_features' in self._dict:
            bg = self._dict['bg_features']
            print(f'[ datasets/buffer ] Found bg_features: shape={bg.shape}, '
                  f'range=[{bg.min():.3f}, {bg.max():.3f}]')

        print(f'[ datasets/buffer ] Loaded {self._count} episodes from {path}')
