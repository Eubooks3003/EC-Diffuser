import numpy as np
import pickle


def _install_numpy_pickle_shim():
    import sys
    import numpy.core as _core
    sys.modules["numpy._core"] = _core
    sys.modules["numpy._core.multiarray"] = _core.multiarray


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

    def load_paths_from_pickles(self, pkls_with_ids, max_demos_per_task=None,
                                 single_view=False):
        """
        Load multiple per-task pkls into a single buffer.

        Each per-task pkl is expected to share the same feature shape (Dtok, K,
        action/gripper/bg dims). Per-task max_path_length may differ; we
        allocate to the global max and pad with zeros (per-episode
        `path_lengths` is the source of truth for valid frames).

        Args:
            pkls_with_ids: list of (pkl_path, task_id_int).
            max_demos_per_task: int or None — keep first N episodes per task.
            single_view: passed through to per-pkl single-view slicing logic.

        Adds a `task_ids` field of shape (n_episodes,) to self._dict.
        """
        _install_numpy_pickle_shim()

        loaded = []  # list of (paths_dict, task_id, n_eps)
        for pkl_path, task_id in pkls_with_ids:
            with open(pkl_path, "rb") as f:
                paths_dict = pickle.load(f)

            paths_dict = self._maybe_apply_single_view(paths_dict, single_view)
            n_eps = paths_dict['observations'].shape[0]
            if max_demos_per_task is not None:
                n_eps = min(n_eps, max_demos_per_task)
            loaded.append((paths_dict, task_id, n_eps))

        # All keys come from the first task (must agree across tasks).
        sample = loaded[0][0]
        keys = [k for k in sample.keys() if k != 'meta']

        # Validate that all tasks have the same set of (non-meta) keys.
        for paths_dict, task_id, _ in loaded[1:]:
            other_keys = {k for k in paths_dict.keys() if k != 'meta'}
            if set(keys) != other_keys:
                raise ValueError(
                    f"[buffer] task_id={task_id} pkl has keys {sorted(other_keys)} "
                    f"but task_id={loaded[0][1]} has {sorted(keys)}"
                )

        # Determine global allocation shape. For 4D obs/goals we also take the
        # max K across tasks (defensive — should be uniform for d0 mimicgen).
        global_T = 0
        global_K = 0
        for paths_dict, _, _ in loaded:
            obs = paths_dict['observations']
            global_T = max(global_T, obs.shape[1])
            if obs.ndim == 4:
                global_K = max(global_K, obs.shape[2])

        total_eps = sum(n for _, _, n in loaded)

        out = {}
        for k in keys:
            arr = sample[k]
            if k == 'path_lengths':
                out[k] = np.zeros(total_eps, dtype=np.int32)
                continue
            shape = list(arr.shape)
            shape[0] = total_eps
            if arr.ndim >= 2:
                shape[1] = global_T
            if k in ('observations', 'goals') and arr.ndim == 4:
                shape[2] = global_K
            out[k] = np.zeros(tuple(shape), dtype=np.float32)
        out['task_ids'] = np.zeros(total_eps, dtype=np.int32)

        offset = 0
        for paths_dict, task_id, n_eps in loaded:
            for k in keys:
                arr = paths_dict[k][:n_eps]
                if k == 'path_lengths':
                    out[k][offset:offset + n_eps] = arr.astype(np.int32)
                elif arr.ndim == 1:
                    out[k][offset:offset + n_eps] = arr.astype(np.float32)
                else:
                    T = arr.shape[1]
                    if k in ('observations', 'goals') and arr.ndim == 4:
                        K = arr.shape[2]
                        out[k][offset:offset + n_eps, :T, :K] = arr.astype(np.float32)
                    else:
                        out[k][offset:offset + n_eps, :T] = arr.astype(np.float32)
            out['task_ids'][offset:offset + n_eps] = int(task_id)
            offset += n_eps

        for k, v in out.items():
            self._dict[k] = v
        self._count = total_eps
        self.keys = keys + ['task_ids']

        if 'gripper_state' in self._dict:
            gs = self._dict['gripper_state']
            print(f'[ datasets/buffer ] Found gripper_state: shape={gs.shape}, '
                  f'range=[{gs.min():.3f}, {gs.max():.3f}]')
        if 'bg_features' in self._dict:
            bg = self._dict['bg_features']
            print(f'[ datasets/buffer ] Found bg_features: shape={bg.shape}, '
                  f'range=[{bg.min():.3f}, {bg.max():.3f}]')
        per_task = np.bincount(self._dict['task_ids'], minlength=int(self._dict['task_ids'].max()) + 1)
        print(f'[ datasets/buffer ] Multitask buffer: {self._count} episodes total, '
              f'per-task counts={per_task.tolist()}, T_alloc={global_T}, K_alloc={global_K}')

    def _maybe_apply_single_view(self, paths_dict, single_view):
        """Mirror the single-view slicing logic from `load_paths_from_pickle`."""
        if not single_view:
            return paths_dict

        meta = paths_dict.get('meta', None)
        if isinstance(meta, dict):
            K_expected = meta.get('K', None) or meta.get('num_entity', None)
        else:
            K_expected = None

        for key in ('observations', 'goals'):
            if key not in paths_dict:
                continue
            val = paths_dict[key]
            if isinstance(val, np.ndarray) and val.ndim == 4:
                E, T, K, D = val.shape
                if K_expected is not None:
                    if K == 2 * K_expected:
                        val = val[:, :, :K_expected, :]
                paths_dict[key] = val

        if 'bg_features' in paths_dict:
            bg = paths_dict['bg_features']
            if isinstance(bg, np.ndarray) and bg.ndim == 3:
                bg_total = bg.shape[-1]
                if bg_total % 2 == 0:
                    paths_dict['bg_features'] = bg[:, :, : bg_total // 2]
        return paths_dict

    def load_paths_from_pickle(self, path, single_view=False):
        _install_numpy_pickle_shim()
        with open(path, "rb") as f:
            paths_dict = pickle.load(f)

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

            # Single-view: slice bg_features to keep only the first view
            if single_view and key == 'bg_features':
                if isinstance(val, np.ndarray) and val.ndim == 3:
                    # val: (E, T, bg_dim) where bg_dim = bg_per_view * num_views
                    bg_total = val.shape[-1]
                    if bg_total % 2 == 0:
                        bg_per_view = bg_total // 2
                        val = val[:, :, :bg_per_view]

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
