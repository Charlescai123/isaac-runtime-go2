# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A simple locomotion task and termination condition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

import os
import sys
import inspect
import dataclasses

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)


@dataclasses.dataclass
class MoveForwardTaskParams:
    move_forward_coeff: float = 1,
    time_step_s: float = 0.001,
    num_action_repeat: float = 33,
    height_fall_coeff: float = 0.24,
    alive_reward: float = 0.1,
    target_vel: float = 1.0,
    check_contact: float = False,  # for collision detection and obstacle avoidance
    target_vel_dir: float = (1, 0),
    energy_weight: float = -0.005


class MoveForwardTask:
    """move forward task."""

    def __init__(self, params):
        """Initializes the task."""
        self.params = params
        self._draw_ref_model_alpha = 1.
        self.move_forward_coeff = params.move_forward_coeff
        self._alive_reward = np.array(params.alive_reward)
        self._time_step = np.array(params.time_step_s)
        self.num_action_repeat = np.array(params.num_action_repeat)
        self.height_fall_coeff = np.array(params.height_fall_coeff)
        self.target_vel = np.array(params.target_vel)
        self.check_contact = np.array(params.check_contact)
        # return
        self.target_vel_dir = np.array(params.target_vel_dir)

    def __call__(self, env):
        return self.reward(env)

    def reset(self, env):
        """Resets the internal state of the task."""
        self._env = env
        self.last_base_pos = env.robot.GetBasePosition()
        self.current_base_pos = self.last_base_pos

    def update(self, env):
        """Updates the internal state of the task."""
        self.last_base_pos = self.current_base_pos
        self.current_base_pos = env.robot.GetBasePosition()

    def done(self, env):
        """Checks if the episode is over."""
        del env
        env = self._env
        root_pos_sim, _ = env._pybullet_client.getBasePositionAndOrientation(env.robot.quadruped)

        rot_quat = env.robot.GetBaseOrientation()
        rot_euler = env.pybullet_client.getEulerFromQuaternion(rot_quat)
        rot_fall = abs(rot_euler[-1]) > 0.5
        height_fall = root_pos_sim[2] < self.height_fall_coeff or root_pos_sim[2] > 0.8

        contact_done = False

        # collision detection for obstacle avoidance
        if self.check_contact:
            contacts = env._pybullet_client.getContactPoints(bodyA=env.robot.quadruped)

            for contact in contacts:
                if contact[2] is env._world_dict["ground"] or \
                        ("terrain" in env._world_dict and
                         contact[2] is env._world_dict["terrain"]):
                    if contact[3] not in env.robot._foot_link_ids:
                        contact_done = True
                        break

            for contact in contacts:
                if contact[2] is not env._world_dict["ground"]:
                    contact_done = True
                    break

            speed = (np.array(self.current_base_pos) - np.array(self.last_base_pos)) / \
                    (self._time_step * self.num_action_repeat)

            contact_done = contact_done and np.linalg.norm(speed) <= 0.05

        terminated = height_fall or rot_fall or contact_done

        return terminated

    def reward(self, env):
        """Get the reward without side effects."""

        del env
        env = self._env

        energy_reward = np.dot(env.robot.GetMotorTorques(), env.robot.GetMotorTorques())
        move_forward_reward = self._calc_reward_root_velocity()
        alive_reward = self._alive_reward
        reward = move_forward_reward * self.move_forward_coeff + energy_reward * self.params.energy_weight + alive_reward

        return reward

    def _calc_reward_root_velocity(self):
        """Get the root velocity reward."""

        # Here we use the average speed calculated
        x_speed = (self.current_base_pos[0] - self.last_base_pos[0]) / (self._time_step * self.num_action_repeat)
        y_speed = (self.current_base_pos[1] - self.last_base_pos[1]) / (self._time_step * self.num_action_repeat)

        xy_speed = np.array([x_speed, y_speed]).reshape(-1)

        along_speed = np.dot(xy_speed, self.target_vel_dir.reshape(-1))

        # here the reward encourages the agent to be equal or greater than the target velocity
        along_speed = np.clip(along_speed, a_min=None, a_max=self.target_vel)

        forward_reward = self.target_vel ** 2 - (along_speed - self.target_vel) ** 2

        return forward_reward

