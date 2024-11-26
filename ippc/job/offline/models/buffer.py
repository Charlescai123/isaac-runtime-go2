import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple, Union

TensorBatch = List[torch.Tensor]


class Trajectory:
    def __init__(self, length, state_dim, action_dim):
        self.length = length
        # self.state = np.zeros((length, state_dim), dtype=np.float32)
        # self.action = np.zeros((length, action_dim), dtype=np.float32)
        # self.reward = np.zeros((length, 1), dtype=np.float32)
        self.state = []
        self.action = []
        self.reward = []

    def add_item(self, input_state, input_action, input_reward, idx):
        self.state.append(input_state)
        self.action.append(input_action)
        self.reward.append(input_reward)

    def get_item(self, idx):
        if idx == len(self.state) - 1:
            next_state = self.state[idx]
        else:
            next_state = self.state[idx + 1]
        return self.state[idx], self.action[idx], self.reward[idx], next_state

    def create_dict(self):
        data = {}
        data["observations"] = np.array(self.state)
        data["actions"] = np.array(self.action)
        data["rewards"] = np.array(self.reward)
        if len(self.state) > 2:
            data["next_observations"] = np.concatenate((np.array(self.state[1:]), np.array([self.state[-1]])), axis=0)
        elif len(self.state) == 2:
            data["next_observations"] = np.concatenate((np.array([self.state[-1]]), np.array([self.state[-1]])), axis=0)
        else:
            data["next_observations"] = np.array([self.state[-1]])
        # for i in range(len(data["next_observations"]) - 1):
        #     data["next_observations"][i] = self.state[i + 1]
        data["terminals"] = np.zeros((len(self.state), 1), dtype=np.float32)
        data["terminals"][-1][0] = 1
        return data


# CORL implementation


class ReplayBuffer:           # randomly select history data (prevent time-dependent data influence)
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        seed: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0
        self.seed = seed
        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_data(self, data):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        np.random.seed(self.seed)
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)    #randomly select one batch
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_state = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_state, dones]
