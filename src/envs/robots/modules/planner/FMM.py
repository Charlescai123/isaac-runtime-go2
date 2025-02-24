"https://github.com/MarkFzp/navigation-locomotion/blob/main/plan_cts/fmm.py"
import time

import numpy as np
from numpy.core.fromnumeric import mean
from numpy.lib import utils
import skfmm
import skimage
from numpy import ma


def get_mask(sx, sy, step_size):
    size = int(step_size) * 2 + 1
    mask = np.zeros((size, size))
    for j in range(size):
        for i in range(size):
            if step_size ** 2 >= ((i + 0.5) - (size // 2 + sx)) ** 2 + ((j + 0.5) - (size // 2 + sy)) ** 2 > (
                    step_size - 1) ** 2:
                mask[j, i] = 1

    # mask[size // 2, size // 2] = 1
    return mask


def get_dist(sx, sy, step_size):
    size = int(step_size) * 2 + 1
    mask = np.zeros((size, size)) + 1e-10
    for j in range(size):
        for i in range(size):
            if ((i + 0.5) - (size // 2 + sx)) ** 2 + ((j + 0.5) - (size // 2 + sy)) ** 2 <= step_size ** 2:
                mask[j, i] = max(5,
                                 (((i + 0.5) - (size // 2 + sx)) ** 2 +
                                  ((j + 0.5) - (size // 2 + sy)) ** 2) ** 0.5)
    return mask


class FMMPlanner:
    def __init__(self, traversable, add_obstacle_dist=True, obstacle_dist_cap=600, obstacle_dist_ratio=1,
                 resolution=0.015):
        self.traversable = traversable
        self.fmm_dist = None
        self.add_obstacle_dist = add_obstacle_dist
        self.obstacle_dist_cap = obstacle_dist_cap
        self.obstacle_dist_ratio = obstacle_dist_ratio
        self.large_value = 1e10
        self.resolution = resolution
        self.grid_offset = np.array([0.5, 0.5])
        self.fovs = np.linspace(-0.5, 0.5, num=5)
        self.small_fovs = np.linspace(-0.2, 0.2, num=5)
        self.speeds = np.arange(0.05, 0.95, 0.05) / self.resolution
        self.conservative_step_size_factor = 0.9

        self.robot_size = 0.27 / self.resolution
        # assert (self.robot_size % 2 == 1)
        self.half_robot_size = int(self.robot_size / 2)
        # self.step_sizes = np.array([10, 16, 22, 28]) # 0.3, 0.48, 0.66, 0.84
        # self.vels =X self._sample_velocity()

    def preprocess_map(self, map, use_config_space, use_gaussian_filter=False):
        map = (map < 50)
        if use_config_space:
            y_size, x_size = map.shape
            padded_map = np.pad(map, self.half_robot_size, 'constant', constant_values=1)
            map = np.min(np.stack(
                [padded_map[y: y + y_size, x: x + x_size] for y in range(self.half_robot_size * 2) for x in
                 range(self.half_robot_size * 2)], axis=-1), axis=-1)

        return map

    def set_goal(self, goal, auto_improve=False, dx=1):
        """Traversable is a binary map with 0-obstacle and 1-free space"""
        traversable_ma = ma.masked_values(self.traversable, 0)
        goal_x, goal_y = int(goal[0]), int(goal[1])

        if self.traversable[goal_x, goal_y] == 0. and auto_improve:
            print(f"Current set goal is obstacle, find another nearest goal...")
            goal_x, goal_y = self._find_nearest_goal([goal_x, goal_y])

        traversable_ma[goal_x, goal_y] = 0

        # For skfmm.distance, 0 should be the goal points (zero-contour) and all values greater than zero are
        # free space with given distance > 0, all values less than zero are also free space with given distance < 0
        masked_distance_map = skfmm.distance(traversable_ma, dx=dx)  # Distance map with the obstacle mask

        dd = ma.filled(masked_distance_map, self.large_value)  # Replace the mask with large value

        if self.add_obstacle_dist:
            helper_planner = FMMPlanner(self.traversable, add_obstacle_dist=False)
            obstacle_dist = helper_planner.set_multi_goal(self.traversable == 0, dx=dx)
            inverse_obstacle_dist = np.minimum(self.obstacle_dist_cap, np.power(obstacle_dist, 2))
            dd += self.obstacle_dist_ratio * (self.obstacle_dist_cap - inverse_obstacle_dist)

        self.fmm_dist = dd

        return dd, masked_distance_map

    def set_multi_goal(self, goal_map, dx=1):
        traversable_ma = ma.masked_values(self.traversable, 0)
        traversable_ma[goal_map == 1] = 0
        dd = skfmm.distance(traversable_ma,
                            dx=dx)  # add_obstacle_dist: outputs a nd array (not masked), with 0 at obstacles
        dd = ma.filled(dd, self.large_value)  # add_obstacle_dist: no effect
        self.fmm_dist = dd
        return dd

    def get_short_term_goal(self, pos, yaw, lin_speed):
        stop = self.is_near_goal(pos_in_map=pos)

        # step_sizes = (lin_speed + self.speeds) / 2 * self.conservative_step_size_factor

        # step_sizes = (lin_speed + np.linspace(np.maximum(lin_speed - 0.4, 0.05),
        #                                       np.minimum(lin_speed + 0.4, 0.95),
        #                                       num=10) / self.resolution) / 2  # [n_step]
        step_sizes = (lin_speed + np.linspace(np.maximum(lin_speed - 0.4, 0.05),
                                              np.minimum(lin_speed + 0.4, 0.95),
                                              num=10)) / 2  # [n_step]

        # print('speed: ', lin_speed)
        # print(f"step_sizes: {step_sizes}")
        fovs_w = yaw + self.fovs
        fovs_w_1d = fovs_w[np.newaxis, :]
        next_pos = (np.stack([np.cos(fovs_w_1d), np.sin(fovs_w_1d)], axis=-1) *
                    step_sizes[:, np.newaxis, np.newaxis] + pos)  # [n_step, n_fov, 2]

        local_vel_angle = fovs_w_1d[..., np.newaxis] + self.small_fovs[np.newaxis, np.newaxis,
                                                       :]  # [1, n_fov, n_small_fov]
        next_vel \
            = (np.stack([np.cos(local_vel_angle), np.sin(local_vel_angle)], axis=-1) *
               step_sizes[:, np.newaxis, np.newaxis, np.newaxis])  # [n_step, n_fov, n_small_fov, 2]

        next_pos = np.stack([next_pos] * len(self.small_fovs), axis=2)

        # print(f"fovs_w: {fovs_w}")
        # print(f"fovs_w_1d: {fovs_w_1d}")
        # print(f"next_pos: {next_pos}")
        # print(f"next_vel: {next_vel}")
        # print(f"local_vel_angle: {local_vel_angle}")

        next_pos_l = next_pos.reshape(-1, 2)
        next_vel_l = next_vel.reshape(-1, 2)

        # print(f"pos: {pos}")
        # print(f"yaw: {yaw}")
        # print(f"lin_speed: {lin_speed}")
        # time.sleep(5)

        return next_pos_l, next_vel_l, stop

    def get_fmm_value(self, coords):
        y, x = coords[:, 0], coords[:, 1]
        y_out = np.logical_or(y < 0, y >= self.fmm_dist.shape[0])
        x_out = np.logical_or(x < 0, x >= self.fmm_dist.shape[1])
        yx_out = np.logical_or(y_out, x_out)
        out_bools = np.stack([yx_out, yx_out], axis=-1)
        out_coords = np.where(yx_out)

        valid_coords = np.where(out_bools, 0, coords.astype(np.int32))
        valid_y, valid_x = valid_coords[:, 0], valid_coords[:, 1]
        fmm_value = self.fmm_dist[(valid_y, valid_x)]
        fmm_value[out_coords] = self.large_value

        return fmm_value

    def find_argmin_traj(self, pos_trajs):
        T = pos_trajs.shape[1]
        flat_pos_trajs = pos_trajs.reshape(-1, 2)
        # print(f"flag_pos_trajs: {flat_pos_trajs}")
        flat_pos_trajs = flat_pos_trajs[::-1]
        fmm_values = self.get_fmm_value(flat_pos_trajs)
        traj_cost = fmm_values.reshape(-1, T)
        argmin_idx = np.argmin(np.sum(traj_cost, axis=-1))
        # print(f"fmm_values: {fmm_values}")
        # print(f"argmin_idx: {argmin_idx}")

        return argmin_idx

    def _find_nearest_goal(self, goal):
        traversable = skimage.morphology.binary_dilation(
            np.zeros(self.traversable.shape),
            skimage.morphology.disk(2)) != True

        traversable = traversable * 1.
        planner = FMMPlanner(traversable)
        planner.set_goal(goal)

        mask = self.traversable

        dist_map = planner.fmm_dist * mask
        dist_map[dist_map == 0] = dist_map.max()

        goal = np.unravel_index(dist_map.argmin(), dist_map.shape)

        return goal

    def is_near_goal(self, pos_in_map, arrival_radius=0.6):
        """Determine whether the goal has been reached
        pos_in_map: The position on map
        arrival_radius: Arrival radius as a circle with unit meter (Threshold for checking goal reaching)
        """
        pos_int = [int(x) for x in pos_in_map]

        # Restrict to the occupancy map district
        pos_int[0] = max(0, min(pos_int[0], self.fmm_dist.shape[0] - 1))
        pos_int[1] = max(0, min(pos_int[1], self.fmm_dist.shape[1] - 1))

        if self.fmm_dist[pos_int[0], pos_int[1]] <= arrival_radius / self.resolution:  # 0.25 m
            return True
        else:
            return False
