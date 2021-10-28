import numpy as np

class ReplayBuffer:
    def __init__(self, max_size, state_dims):
        self.max_size = max_size

        self.states = np.zeros((self.max_size, *state_dims), dtype='float32')
        self.next_states = np.zeros((self.max_size, *state_dims), dtype='float32')
        self.actions = np.zeros(self.max_size, dtype='int32')
        self.rewards = np.zeros(self.max_size, dtype='int32')
        self.terminates = np.zeros(self.max_size, dtype='bool')

        self.internal_idx = 0

    def store(self, state, action, reward, next_state, done):
        buff_idx = self.internal_idx % self.max_size

        self.states[buff_idx] = state
        self.next_states[buff_idx] = next_state
        self.actions[buff_idx] = action
        self.rewards[buff_idx] = reward
        self.terminates[buff_idx] = done

        self.internal_idx += 1

    def __len__(self):
        return self.internal_idx if self.internal_idx < self.max_size else self.max_size

    def get_random_replays(self, size):
        if size > self.max_size:
            raise ValueError(f'size ({size}) is greater than max_size ({self.max_size})')
        
        idxs = np.random.choice(self.internal_idx if self.internal_idx < self.max_size else self.max_size, size)

        return self.states[idxs], self.actions[idxs], self.rewards[idxs], self.next_states[idxs], self.terminates[idxs]
