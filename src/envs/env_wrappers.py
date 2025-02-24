"""Environment wrappers for normalizing RL environments."""
import time

from isaacgym.torch_utils import to_torch
import torch


class AttributeModifier:
    def __getattr__(self, name):
        return getattr(self._env, name)

    def set_attribute(self, name, value):
        set_attr = getattr(self._env, 'set_attribute', None)
        if callable(set_attr):
            self._env.set_attribute(name, value)
        else:
            setattr(self._env, name, value)

    @property
    def episode_length_buf(self):
        return self._env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, new_length: torch.Tensor):
        self._env.episode_length_buf = new_length


class RangeNormalize(AttributeModifier):
    def __init__(self, env):
        self._env = env
        self._device = self._env.device

    @property
    def observation_space(self):
        low, high = self._env.observation_space
        return -torch.ones_like(low), torch.ones_like(low)

    @property
    def action_space(self):
        low, high = self._env.action_space
        return -torch.ones_like(low), torch.ones_like(low)

    def step(self, drl_action_norm):
        # magnitude = to_torch([4, 4, 2, 8, 8, 4], device=self._device) * 0.5  # 6 dims
        # magnitude = to_torch([4, 4, 2, 8, 2, 0.], device=self._device) * 2 # 6 dims
        # magnitude = np.array([2, 1, 2, 2, 1, 0.5]) * 2  # 6 dims
        # drl_action_norm *= magnitude
        drl_action = self._denormalize_action(drl_action_norm)      # Denormalize drl action from NN
        observ, privileged_obs, action, reward, done, info = self._env.step(drl_action)
        observ = self._normalize_observ(observ)     # Normalized observation
        return observ, privileged_obs, action, reward, done, info

    def reset(self):
        observ, privileged_obs = self._env.reset()
        observ = self._normalize_observ(observ)
        return observ, privileged_obs

    def _denormalize_action(self, action):
        """denormalize from [-1, 1]"""
        min_ = self._env.action_space[0]
        max_ = self._env.action_space[1]
        action = (action + 1) / 2 * (max_ - min_) + min_
        return action

    def _normalize_observ(self, observ):
        """normalize to [-1, 1]"""
        min_ = self._env.observation_space[0]
        max_ = self._env.observation_space[1]
        observ = 2 * (observ - min_) / (max_ - min_) - 1
        return observ

    def get_observations(self):
        obs = self._env.get_observations()
        return self._normalize_observ(obs)


class ClipAction(AttributeModifier):
    """Clip out of range actions to the action space of the environment."""

    def __init__(self, env):
        self._env = env

    def step(self, action):
        action_space = self._env.action_space
        action = torch.clip(action, action_space[0], action_space[1])
        return self._env.step(action)
