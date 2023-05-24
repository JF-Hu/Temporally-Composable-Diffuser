import numpy as np

def atleast_2d(x):
    while x.ndim < 2:
        x = np.expand_dims(x, axis=-1)
    return x

class ReplayBuffer:

    def __init__(self, max_n_episodes, max_path_length, termination_penalty):
        self._dict = {
            'path_lengths': np.zeros(max_n_episodes, dtype=np.int),
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
                if k != 'path_lengths'}.items()

    def _allocate(self, key, array):
        assert key not in self._dict
        dim = array.shape[-1]
        shape = (self.max_n_episodes, self.max_path_length, dim)
        self._dict[key] = np.zeros(shape, dtype=np.float32)
        # print(f'[ utils/mujoco ] Allocated {key} with size {shape}')

    def path_padding(self, path, history_length, head_padding=True):
        for key, val in path.items():
            if len(np.shape(path[key])) == 1:
                padding_array = np.zeros((history_length - 1,))
            else:
                padding_array = np.zeros((history_length - 1, np.shape(path[key])[-1]))
            if head_padding:
                path[key] = np.concatenate([padding_array, path[key]], axis=0)
            else:
                path[key] = np.concatenate([path[key], padding_array], axis=0)
        return path

    def add_path(self, path, pos_enc=None, history_length=1, discounts=None):
        if history_length > 1:
            # todo if history_length > 1, which means that we needs the DM to consider the history information
            path = self.path_padding(path=path, history_length=history_length, head_padding=True)

        path_length = len(path['observations'])
        assert path_length <= self.max_path_length

        # ## penalize early termination
        # if path['terminals'].any() and self.termination_penalty is not None:
        #     assert not path['timeouts'].any(), 'Penalized a timeout episode for early termination'
        #     path['rewards'][path_length - 1] += self.termination_penalty

        if discounts is not None:
            path['discounted_returns'] = np.zeros(shape=np.shape(path['rewards']))
            for idx in range(path_length):
                path['discounted_returns'][idx] = np.sum(path['rewards'][idx:] * np.reshape(discounts[:path_length-idx], np.shape(path['rewards'][idx:])))

        # path['fft_observations'] = np.fft.fft(path['observations'], axis=0)

        if path['terminals'].any():
            assert (path['terminals'][-1] == True) and (not path['terminals'][:-1].any())

        ## if first path added, set keys based on contents
        self._add_keys(path)

        ## add tracked keys in path
        for key in self.keys:
            array = atleast_2d(path[key])
            if key not in self._dict: self._allocate(key, array)
            self._dict[key][self._count, :path_length] = array

            if pos_enc is not None:
                if key == 'observations' or key == 'next_observations':
                    self._dict[key][self._count, :] += pos_enc

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
            self._dict[key] = self._dict[key][:self._count]
        self._add_attributes()
        print(f'[ datasets/buffer ] Finalized replay buffer | {self._count} episodes')
