import gymnasium
import numpy as np
import os
import sys
import inspect
from quad_gym.env.robots import a1
from quad_gym.wrapper.base_wrapper import BaseWrapper

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)


class ActionRestrain(gymnasium.ActionWrapper):
    # Current for POSITION only
    def __init__(self, env, clip_num):
        super().__init__(env)

        self.base_angle = np.array(list(a1.INIT_MOTOR_ANGLES))
        self.clip_num = clip_num
        if isinstance(self.clip_num, list):
            self.clip_num = np.array(self.clip_num)
            assert len(clip_num) == np.prod(self.base_angle.shape)

        self.ub = self.base_angle + self.clip_num
        self.lb = self.base_angle - self.clip_num
        self.action_space = gymnasium.spaces.Box(self.lb, self.ub)

    def action(self, action):
        return np.clip(action, self.lb, self.ub)


class MPCActionRestrain(gymnasium.ActionWrapper):

    def __init__(self, env, clip_num):
        super().__init__(env)

        self.clip_num = clip_num
        if isinstance(self.clip_num, list):
            self.clip_num = np.array(self.clip_num)
            assert len(clip_num) == 2

        self.ub = self.clip_num
        self.lb = - self.clip_num
        self.action_space = gymnasium.spaces.Box(self.lb, self.ub)

    def action(self, action):
        return np.clip(action, self.lb, self.ub)


class ActionResidual(gymnasium.ActionWrapper):
    # Current for POSITION only
    def __init__(self, env, clip_num):
        super().__init__(env)
        self.clip_num = clip_num
        self.base_angle = np.array(a1.INIT_MOTOR_ANGLES)
        self.ub = np.ones_like(self.base_angle) * self.clip_num
        self.lb = -self.ub
        self.action_space = gymnasium.spaces.Box(self.lb, self.ub)

    def action(self, action):
        current_angles = self.robot.GetMotorAngles()
        biased_action = np.clip(action,
                                self.lb, self.ub
                                ) + current_angles
        return biased_action


class DiagonalAction(gymnasium.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.lb = np.split(self.env.action_space.low, 2)[0]
        self.ub = np.split(self.env.action_space.high, 2)[0]
        self.action_space = gymnasium.spaces.Box(self.lb, self.ub)

    def action(self, action):
        right_act, left_act = np.split(action, 2)
        act = np.concatenate(
            [right_act, left_act, left_act, right_act]
        )
        return act


class RandoDirWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env, dir_update_interval=None):
        super().__init__(env)

        self.observation_space = gymnasium.spaces.Box(
            np.concatenate([[0, 0], self.env.observation_space.low]),
            np.concatenate([[1, 1], self.env.observation_space.high])
        )
        self.current_angle = 0
        self.current_vec = np.array([
            np.cos(self.current_angle),
            np.sin(self.current_angle)
        ])
        self.dir_update_interval = dir_update_interval
        self.time_count_randdir = 0

    def observation(self, observation):
        self.time_count_randdir += 1
        if self.dir_update_interval is not None and \
                self.time_count_randdir % self.dir_update_interval == 0:
            self.current_angle = np.random.uniform(
                low=-np.pi / 2,
                high=np.pi / 2
            )
            self.current_vec = np.array([
                np.cos(self.current_angle),
                np.sin(self.current_angle)
            ])
            self.env.task.target_vel_dir = self.current_vec

        obs = np.concatenate([self.current_vec, observation])

        return obs

    def reset(self):
        self.time_count_randdir = 0
        self.current_angle = np.random.uniform(
            low=-np.pi / 2,
            high=np.pi / 2
        )
        self.current_vec = np.array([
            np.cos(self.current_angle),
            np.sin(self.current_angle)
        ])
        self.env.task.target_vel_dir = self.current_vec
        return super().reset()


class NormAct(gymnasium.ActionWrapper, BaseWrapper):
  """
  Normalized Action      => [ -1, 1 ]
  """

  def __init__(self, env):
    super(NormAct, self).__init__(env)
    ub = np.ones(self.env.action_space.shape)
    self.action_space = gymnasium.spaces.Box(-1 * ub, ub)
    self.lb = self.env.action_space.low
    self.ub = self.env.action_space.high

  def action(self, action):
    action = np.tanh(action)
    scaled_action = self.lb + (action + 1.) * 0.5 * (self.ub - self.lb)
    return np.clip(scaled_action, self.lb, self.ub)