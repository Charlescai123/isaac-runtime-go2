import time

import torch
import tensorflow as tf
from typing import Callable, Dict, Generator, Tuple, Optional

from rsl_rl.storage.storage import Dataset, Storage, Transition


class ReplayStorageTF(Storage):
    def __init__(self, environment_count: int, max_size: int, device: str = "cpu", initial_size: int = 0) -> None:
        self._env_count = environment_count
        self.initial_size = initial_size // environment_count
        self.max_size = max_size

        self.device = device

        self._register_serializable("max_size", "initial_size")

        self._idx = 0
        self._full = False
        self._initialized = initial_size == 0
        self._data = {}

        self._processors: Dict[Tuple[Callable, Callable]] = {}

    @property
    def max_size(self):
        return self._size * self._env_count

    @max_size.setter
    def max_size(self, value):
        self._size = value // self._env_count

        assert self.initial_size <= self._size

    def _add_item(self, name: str, value: torch.Tensor) -> None:
        """Adds a transition item to the storage.

        Args:
            name (str): The name of the item.
            value (torch.Tensor): The value of the item.
        """
        value = self._process(name, value)
        # print(f"name: {name}")
        # print(f"name: {value.dtype}")
        # print(f"value.shape[1:]: {value.shape[1:]}")
        # print(f"value.shape[1:]: {self._size * self._env_count}")
        if name not in self._data:
            if self._full or self._idx != 0:
                raise ValueError(f'Tried to store invalid transition data for "{name}".')
            self._data[name] = tf.Variable(
                tf.zeros(shape=(self._size * self._env_count, *value.shape[1:]), dtype=value.dtype))

        start_idx = self._idx * self._env_count
        end_idx = (self._idx + 1) * self._env_count
        # print(f"self._data[name][start_idx:end_idx]: {self._data[name][start_idx:end_idx].shape}")
        # print(f"value: {value.shape}")
        self._data[name][start_idx:end_idx].assign(value)

    def _process(self, name: str, value: torch.Tensor) -> torch.Tensor:
        if name not in self._processors:
            return value

        for process, _ in self._processors[name]:
            if process is None:
                continue

            value = process(value)

        return value

    def _process_undo(self, name: str, value: torch.Tensor) -> torch.Tensor:
        if name not in self._processors:
            return value

        for _, undo in reversed(self._processors[name]):
            if undo is None:
                continue

            value = undo(value)

        return value

    def append(self, dataset: Dataset) -> None:
        """Appends a dataset of transitions to the storage.

        Args:
            dataset (Dataset): The dataset of transitions.
        """
        for transition in dataset:
            for name, value in transition.items():
                self._add_item(name, value)

            self._idx += 1

            if self._idx >= self.initial_size:
                self._initialized = True

            if self._idx == self._size:
                self._full = True
                self._idx = 0

        # print(f"self._idx: {self._idx}")
        # print(f"self._size: {self._size}")

    def batch_generator(self, batch_size: int, batch_count: int) -> Generator[Transition, None, None]:
        """Returns a generator that yields batches of transitions.

        Args:
            batch_size (int): The size of the batches.
            batch_count (int): The number of batches to yield.
        Returns:
            A generator that yields batches of transitions.
        """
        assert self._full or self._idx > 0

        if not self._initialized:
            return

        max_idx = self._env_count * (self._size if self._full else self._idx)

        for _ in range(batch_count):
            # batch_idx = torch.randint(high=max_idx, size=(batch_size,))
            batch_idx = tf.random.uniform(shape=(batch_size,), maxval=max_idx, dtype=tf.int32)

            batch = {}
            for key, value in self._data.items():
                # print(f"value: {value}")
                # print(f"batch_idx: {batch_idx}")
                # print(f"value[batch_idx]: {value[batch_idx]}")
                batch[key] = self._process_undo(key, tf.gather(value, batch_idx))

            yield batch

    def register_processor(self, key: str, process: Callable, undo: Optional[Callable] = None) -> None:
        """Registers a processor for a transition item.

        The processor is called before the item is stored in the storage. The undo function is called when the item is
        retrieved from the storage. The undo function is called in reverse order of the processors so that the order of
        the processors does not matter.

        Args:
            key (str): The name of the transition item.
            process (Callable): The function to process the item.
            undo (Optional[Callable], optional): The function to undo the processing. Defaults to None.
        """
        if key not in self._processors:
            self._processors[key] = []

        self._processors[key].append((process, undo))

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def sample_count(self) -> int:
        """Returns the number of individual transitions stored in the storage."""
        transition_count = self._size * self._env_count if self._full else self._idx * self._env_count

        return transition_count

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)
