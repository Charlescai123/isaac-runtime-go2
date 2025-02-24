import enum
import time

import torch
import logging
import numpy as np
from typing import Any
import isaacgym
from isaacgym import gymtorch, gymutil, gymapi
from isaacgym.torch_utils import to_torch

logger = logging.getLogger(__name__)


class ActionMode(enum.Enum):
    STUDENT = 0
    TEACHER = 1
    UNCERTAIN = 2


def energy_value(state: Any, p_mat: np.ndarray) -> int:
    """
    Get energy value represented by s^T @ P @ s -> return a value
    """
    # print(f"state is: {state}")
    # print(f"p_mat: {p_mat}")
    return np.squeeze(np.asarray(state).T @ p_mat @ state)


def energy_value_2d(state: torch.Tensor, p_mat: torch.Tensor) -> torch.Tensor:
    """
    Get energy value represented by s^T @ P @ s (state is a 2d vector) -> return a 1d array
    """
    # print(f"state is: {state}")
    # print(f"p_mat: {p_mat}")
    sp = torch.matmul(state, p_mat)

    return torch.sum(sp * state, dim=1)


def check_safety(error_state: torch.Tensor, safety_subset: torch.Tensor):
    """Check current safety condition w.r.t the system's error states"""
    is_activate = torch.abs(error_state[:, 2:]) >= safety_subset
    return is_activate.any(dim=1)


def generate_seed_sequence(seed, num_seeds):
    np.random.seed(seed)
    return np.random.randint(0, 100, size=num_seeds)


class RobotPusher:
    def __init__(self, robot, sim, viewer, num_envs, device="cuda"):
        self._sim = sim
        self._gym = gymapi.acquire_gym()
        self._robot = robot
        self._viewer = viewer
        self._num_envs = num_envs
        self._device = device

        # Push statistics
        self.push_enable = False
        self.push_cnt = 0
        self.begin_push_step = 98

        # Push Indicator
        self.indicator_flag = False
        self.indicator_cnt = 0
        self.indicator_max = 50
        random_push_sequence = generate_seed_sequence(seed=1, num_seeds=100)

        # For backwards
        # push_delta_vel_list_x = [-0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25]
        # push_delta_vel_list_y = [-0.62, 1.25, -0.7, 0.6, -0.55, 0.6, -0.6]
        # push_delta_vel_list_z = [-0.72, -0.72, -0.72, -0.72, -0.72, -0.72, -0.72]
        # push_interval = np.array([300, 450, 620, 750, 820, 950, 1050, 1200]) - 1

        # For forward
        self._push_delta_vel_list_x = to_torch([0.25, 0.25, 0.25, 0.25, 0.3, 0.25, 0.25], device=device)
        self._push_delta_vel_list_y = to_torch([-0.55, 0.65, -0.6, 0.6, -0.7, 0.7, -0.6], device=device)
        self._push_delta_vel_list_z = to_torch([-0.762, -0.7, -0.72, -0.72, -0.72, -0.72, -0.72], device=device)

        # Push interval
        self._push_interval = to_torch([200, 380, 620, 750, 850, 1000, 1050, 1200], dtype=torch.int, device=device) - 1

    def monitor_push(self, step_cnt, env_ids):
        if step_cnt > self.begin_push_step and step_cnt == self._push_interval[self.push_cnt]:

            if self.push_enable:
                print(f"cnt is: {step_cnt}, pushing the robot now")
                self._push_robot_idx(env_ids)
                # self._push_robots()

                self.indicator_flag = True
                return True
        return False

    def _push_robot_idx(self, env_ids):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity.
        """
        env_ids_int32 = self._robot._num_actor_per_env * env_ids.to(dtype=torch.int32)

        curr_vx = self._robot._root_states[env_ids, 7]
        curr_vy = self._robot._root_states[env_ids, 8]
        curr_vz = self._robot._root_states[env_ids, 9]
        print(f"self.robot._root_states: {self._robot._root_states}")

        delta_x = self._push_delta_vel_list_x[self.push_cnt]
        delta_y = self._push_delta_vel_list_y[self.push_cnt]
        delta_z = self._push_delta_vel_list_z[self.push_cnt]

        print(f"delta_x: {delta_x}")
        print(f"delta_y: {delta_y}")
        print(f"delta_z: {delta_z}")
        # time.sleep(2)

        vel_after_push_x = curr_vx + delta_x
        vel_after_push_y = curr_vy + delta_y
        vel_after_push_z = curr_vz + delta_z

        print(f"vel_after_push_x: {vel_after_push_x}")
        print(f"vel_after_push_y: {vel_after_push_y}")
        print(f"vel_after_push_z: {vel_after_push_z}")

        # Turn on the push indicator for viewing
        self.draw_push_indicator(env_ids=env_ids, target_pos=[delta_x, delta_y, delta_z])
        print(f"push robot before: {self._robot._root_states}")

        actor_root_state = self._gym.acquire_actor_root_state_tensor(self._sim)
        actor_root_state = gymtorch.wrap_tensor(actor_root_state)
        actor_root_state[env_ids_int32, 7] = vel_after_push_x
        actor_root_state[env_ids_int32, 8] = vel_after_push_y
        actor_root_state[env_ids_int32, 9] = vel_after_push_z

        self._gym.set_actor_root_state_tensor_indexed(self._sim,
                                                      gymtorch.unwrap_tensor(actor_root_state),
                                                      gymtorch.unwrap_tensor(env_ids_int32),
                                                      1)
        # print(f"push robot after: {self._robot._root_states}")
        self.push_cnt += 1

    def draw_push_indicator(self, env_ids, target_pos=[1., 0., 0.]):
        """Draw the line indicator for pushing the robot"""

        sphere_geom_arrow = gymutil.WireframeSphereGeometry(0.01, 50, 50, None, color=(1, 0., 0.))
        pose_robot = self._robot._root_states[env_ids, :3].squeeze(dim=0).cpu().numpy()
        print(f"pose_robot: {pose_robot}")
        self.target_pos_rel = to_torch([target_pos], device=self._device)
        for i in range(5):
            norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
            target_vec_norm = self.target_pos_rel / (norm + 1e-5)
            print(f"norm: {norm}")
            print(f"target_vec_norm: {target_vec_norm}")
            # pose_arrow = pose_robot[:3] + 0.1 * (i + 3) * target_vec_norm[:self._num_envs, :3].cpu().numpy()

            xy = pose_robot[:2] + 0.08 * (i + 3) * target_vec_norm[env_ids, :2].cpu().numpy()
            z = pose_robot[2] + 0.03 * (i + 3) * target_vec_norm[env_ids, 2].cpu().numpy()
            print(f"xy: {xy}")
            print(f"xy: {z}")
            pose_arrow = np.hstack((xy.squeeze(), z))
            # pose_arrow = pose_arrow.squeeze()
            print(f"pose_arrow: {pose_arrow}")
            pose = gymapi.Transform(gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_arrow[2]), r=None)
            print(f"pose: {pose}")
            for i in range(len(env_ids)):
                idx = env_ids[i].item()
                gymutil.draw_lines(sphere_geom_arrow, self._gym, self._viewer, self._robot._envs[idx], pose)

        sphere_geom_arrow = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(0, 1, 0.5))
        # for i in range(5):
        #     norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)
        #     target_vec_norm = self.next_target_pos_rel / (norm + 1e-5)
        #     pose_arrow = pose_robot[:2] + 0.2 * (i + 3) * target_vec_norm[self.lookat_id, :2].cpu().numpy()
        #     pose = gymapi.Transform(gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_robot[2]), r=None)
        #     gymutil.draw_lines(sphere_geom_arrow, self.gym, self.viewer, self.envs[self.lookat_id], pose)

    def clear_indicator(self):
        self._gym.clear_lines(self._viewer)
        self.indicator_cnt = 0
        self.indicator_flag = False
