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

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from typing import List

import numpy as np

import os
import sys
import inspect
import dataclasses
from dataclasses import field

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)


@dataclasses.dataclass
class GoalTaskParams:
    target_vel: float = 1.0
    z_constrain: bool = False
    other_direction_penalty: float = 0
    z_penalty: float = 0
    orientation_penalty: float = 0
    time_step_s: float = 0.001
    num_action_repeat: int = 33
    height_fall_coeff: float = 0.24
    alive_reward: float = 0.1
    fall_reward: float = 0
    check_contact: bool = False
    target_vel_dir: List[float] = field(default=(1.0, 0))
    goal_coeff: float = 10
    goal_pose: List[float] = field(default=(1.0, 1.0, 1.0))
    subgoal: bool = False


class GoalTask:
    """
    move to goal task.
    todo think about how to set the goal properly and how to measure the goal on real robot
    """

    def __init__(self, params):
        """Initializes the task."""
        self._draw_ref_model_alpha = 1.
        self.subgoal = params.subgoal
        # self.energy_weight = -0.01
        self.goal_coeff = params.goal_coeff
        self.energy_weight = -0.005
        self.move_forward_coeff = 1
        self._ref_model = -1
        self._alive_reward = params.alive_reward
        self.fall_reward = params.fall_reward
        self._time_step = params.time_step_s
        self.num_action_repeat = params.num_action_repeat
        self.z_constrain = params.z_constrain
        self.other_direction_penalty = params.other_direction_penalty
        self.z_penalty = params.z_penalty
        self.init_orientation = np.array([0, 0, 0, 1])
        self.orientation_penalty = params.orientation_penalty
        self.height_fall_coeff = params.height_fall_coeff
        self.target_vel = params.target_vel
        self.check_contact = params.check_contact
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
        """ Here is the checking for the termination condition.
        For the termination case, we don't use the return at the next time step (Not Bootstrapping)
        For the truncation case, we use the return at the next time step (Bootstrapping)

        In this quadruped robot case, if we are interested in continuous control tasks of infinite horizon, then the
        termination is the robot falls down or the robot is out of the boundary.
        truncation is the time_limit we set for creating the new episode

        """
        del env
        env = self._env
        pyb = env._pybullet_client

        root_pos_sim, root_rot_sim = pyb.getBasePositionAndOrientation(env.robot.quadruped)

        rot_quat = env.robot.GetBaseOrientation()
        rot_mat = env.pybullet_client.getMatrixFromQuaternion(rot_quat)

        rot_fall = rot_mat[-1] < 0.6
        height_fall = root_pos_sim[2] < self.height_fall_coeff
        if self.z_constrain:
            height_fall = root_pos_sim[2] < self.height_fall_coeff or root_pos_sim[2] > 0.8

        contact_done = False

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

        termination = height_fall or rot_fall or contact_done

        return termination

    def reward(self, env):
        """Get the reward without side effects."""
        del env

        env = self._env
        energy_reward = np.dot(
            env.robot.GetMotorTorques(),
            env.robot.GetMotorTorques()
        ) * self._time_step

        move_forward_reward = self._calc_reward_root_velocity()
        alive_reward = self._alive_reward
        orientation_reward = self._calc_reward_rotation()
        goal_reward = self._calc_reward_goal_dist()
        subgoal_reward = self._calc_reward_subgoal() if self.subgoal else 0
        reward = goal_reward * self.goal_coeff + move_forward_reward * self.move_forward_coeff + \
                 energy_reward * self.energy_weight - \
                 self.orientation_penalty * orientation_reward + \
                 alive_reward + subgoal_reward
        termination = self.done(env)
        if termination:
            reward += self.fall_reward  # penalty for falling down
        return reward

    def _get_pybullet_client(self):
        """Get bullet client from the environment"""
        return self._env._pybullet_client

    def _calc_reward_goal_dist(self):
        env = self._env
        last_dist = np.linalg.norm(
            np.array(env._world_dict["goal_pos"]) - self.last_base_pos)
        current_dist = np.linalg.norm(
            np.array(env._world_dict["goal_pos"]) - self.current_base_pos)
        reward = (last_dist - current_dist) / \
                 (self._time_step * self.num_action_repeat)
        return reward

    def _calc_reward_subgoal(self):
        subgoal_pos = self._env.world_dict['subgoals']
        reward = 0
        for i, pos in enumerate(subgoal_pos):
            if np.linalg.norm(self.current_base_pos - np.array(pos)) < 1.0:
                if not self._env.world_dict['subgoals_achieved'][i]:
                    self._env.world_dict['subgoals_achieved'][i] = True
                    reward += 5
        return reward

    def _calc_reward_root_velocity(self):
        """Get the root velocity reward."""
        env = self._env
        robot = env.robot
        sim_model = robot.quadruped

        pyb = self._get_pybullet_client()

        root_vel_sim, _ = pyb.getBaseVelocity(sim_model)
        root_vel_sim = np.array(root_vel_sim)

        x_speed = (self.current_base_pos[0] - self.last_base_pos[0]
                   ) / (self._time_step * self.num_action_repeat)
        y_speed = (self.current_base_pos[1] - self.last_base_pos[1]
                   ) / (self._time_step * self.num_action_repeat)
        z_speed = (self.current_base_pos[2] - self.last_base_pos[2]
                   ) / (self._time_step * self.num_action_repeat)

        xy_speed = np.array([x_speed, y_speed])

        xy_speed = np.linalg.norm(xy_speed)
        xy_speed = np.clip(
            xy_speed, a_min=None, a_max=self.target_vel
        )
        along_reward = self.target_vel ** 2 - (
                xy_speed - self.target_vel
        ) ** 2

        forward_reward = along_reward - self.z_penalty * (z_speed ** 2)

        # y_reward = self.target_vel ** 2 - (y_speed - self.target_vel) ** 2

        # forward_reward = y_reward - \
        #   self.other_direction_penalty * np.abs(x_speed) - \
        #   self.other_direction_penalty * np.abs(z_speed)

        # print("Y_Rew:{:.4f}, Z_Rew:{:.4f}".format(
        #   -self.other_direction_penalty * y_speed,
        #   -self.other_direction_penalty * z_speed
        # ))

        return forward_reward

    def _calc_reward_rotation(self):
        env = self._env
        pyb = self._get_pybullet_client()

        rot_quat = env.robot.GetBaseOrientation()

        if self.init_orientation is None:
            return 0
        # Norm of displacement vector
        rot_reward = np.sum(
            (self.init_orientation - np.array(rot_quat)) ** 2)  # * self.num_action_repeat
        return rot_reward
