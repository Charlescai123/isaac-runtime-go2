from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
# import torch
from typing import Any, Callable, Dict, Tuple, Union
import tensorflow as tf

from rsl_rl.env import VecEnv
from rsl_rl.storage.storage import Dataset
from rsl_rl.utils.benchmarkable import Benchmarkable
from rsl_rl.utils.serializable import Serializable
from rsl_rl.utils.utils import environment_dimensions


class AgentTF(ABC, Benchmarkable, Serializable):
    def __init__(
        self,
        env: VecEnv,
        action_max: float = np.inf,
        action_min: float = -np.inf,
        benchmark: bool = False,
        device: str = "cpu",
        gamma: float = 0.99,
    ):
        """Creates an agent.

        Args:
            env (VecEnv): The environment of the agent.
            action_max (float): The maximum action value.
            action_min (float): The minimum action value.
            benchmark (bool): Whether to benchmark runtime.
            device (str): The device to use for computation.
            gamma (float): The environment discount factor.
        """
        super().__init__()
        self.env = env
        self.device = device
        self.storage = None

        self._action_max = action_max
        self._action_min = action_min
        self._discount_factor = gamma

        self._register_serializable("_action_max", "_action_min", "_discount_factor")

        dimensions = environment_dimensions(self.env)
        self._action_size = dimensions["actions"]

        self._register_serializable("_action_size")

        if self._action_min > -np.inf and self._action_max < np.inf:
            self._rand_scale = self._action_max - self._action_min
            self._rand_offset = self._action_min
        else:
            self._rand_scale = 2.0
            self._rand_offset = -1.0

        self._bm_toggle(benchmark)

    @abstractmethod
    def draw_actions(
        self, obs: tf.Tensor, env_info: Dict[str, Any]
    ) -> Tuple[tf.Tensor, Union[Dict[str, tf.Tensor], None]]:
        """Draws actions from the action space.

        Args:
            obs (tf.Tensor): The observations for which to draw actions.
            env_info (Dict[str, Any]): The environment information for the observations.
        Returns:
            A tuple containing the actions and the data dictionary.
        """
        pass

    def draw_random_actions(
        self, obs: tf.Tensor, env_info: Dict[str, Any]
    ) -> Tuple[tf.Tensor, Union[Dict[str, tf.Tensor], None]]:
        """Draws random actions from the action space.

        Args:
            obs (tf.Tensor): The observations to include in the data dictionary.
            env_info (Dict[str, Any]): The environment information to include in the data dictionary.
        Returns:
            A tuple containing the random actions and the data dictionary.
        """
        actions = self._process_actions(
            self._rand_offset + self._rand_scale * tf.rand(self.env.num_envs, self._action_size)
        )

        return actions, {}

    @abstractmethod
    def eval_mode(self) -> AgentTF:
        """Sets the agent to evaluation mode."""
        return self

    @abstractmethod
    def export_onnx(self) -> Tuple[tf.Module, tf.Tensor, Dict]:
        """Exports the agent's policy network to ONNX format.

        Returns:
            A tuple containing the ONNX model, the input arguments, and the keyword arguments.
        """
        pass

    @property
    def gamma(self) -> float:
        return self._discount_factor

    @abstractmethod
    def get_inference_policy(self, device: str = None) -> Callable:
        """Returns a function that computes actions from observations without storing gradients.

        Args:
            device (tf.device): The device to use for inference.
        Returns:
            A function that computes actions from observations.
        """
        pass

    @property
    def initialized(self) -> bool:
        """Whether the agent has been initialized."""
        return self.storage.initialized

    @abstractmethod
    def process_transition(
        self,
        observations: tf.Tensor,
        environement_info: Dict[str, Any],
        actions: tf.Tensor,
        rewards: tf.Tensor,
        next_observations: tf.Tensor,
        next_environment_info: tf.Tensor,
        dones: tf.Tensor,
        data: Dict[str, tf.Tensor],
    ) -> Dict[str, tf.Tensor]:
        """Processes a transition before it is added to the replay memory.

        Args:
            observations (tf.Tensor): The observations from the environment.
            environment_info (Dict[str, Any]): The environment information.
            actions (tf.Tensor): The actions computed by the actor.
            rewards (tf.Tensor): The rewards from the environment.
            next_observations (tf.Tensor): The next observations from the environment.
            next_environment_info (Dict[str, Any]): The next environment information.
            dones (tf.Tensor): The done flags from the environment.
            data (Dict[str, tf.Tensor]): Additional data to include in the transition.
        Returns:
            A dictionary containing the processed transition.
        """
        pass

    @abstractmethod
    def register_terminations(self, terminations: tf.Tensor) -> None:
        """Registers terminations with the actor critic agent.

        Args:
            terminations (tf.Tensor): A tensor of indicator values for each environment.
        """
        pass

    @abstractmethod
    def to(self, device: str) -> AgentTF:
        """Transfers agent parameters to device."""
        self.device = device

        return self

    @abstractmethod
    def train_mode(self) -> AgentTF:
        """Sets the agent to training mode."""
        return self

    @abstractmethod
    def update(self, dataset: Dataset) -> Dict[str, Union[float, tf.Tensor]]:
        """Updates the agent's parameters.

        Args:
            dataset (Dataset): The dataset from which to update the agent.
        Returns:
            A dictionary containing the loss values.
        """
        pass

    def _process_actions(self, actions: tf.Tensor) -> tf.Tensor:
        """Processes actions produced by the agent.

        Args:
            actions (tf.Tensor): The raw actions.
        Returns:
            A tf.Tensor containing the processed actions.
        """
        # actions = actions.reshape(-1, self._action_size)
        # actions = actions.clamp(self._action_min, self._action_max)
        # actions = actions.to(self.device)
        actions = tf.reshape(actions, (-1, self._action_size))
        actions = tf.clip_by_value(actions, self._action_min, self._action_max)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)

        return actions
