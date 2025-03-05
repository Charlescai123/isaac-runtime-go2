import math

import torch

from src.physical_design import MATRIX_P


class Go2Rewards:
    """Set of rewards for Go2"""

    def __init__(self, env, reward_cfg):
        self._env = env
        self._config = reward_cfg
        self._robot = self._env.robot
        self._gait_generator = self._env.gait_generator
        self._num_envs = self._env.num_envs
        self._device = self._env.device

        self._prepare_reward_fn()

    def _prepare_reward_fn(self):
        self.episode_sums = {
            name: torch.zeros(self._env.num_envs, dtype=torch.float, device=self._device)
            for name in self._config.scales.keys()}

    def compute_reward(self, err_prev, err_next, is_fail, delta_action):
        """The final output reward"""
        tot_rwd = 0
        for rwd_fn_name, scale in self._config.scales.items():
            reward_fn = getattr(self, rwd_fn_name + '_reward')
            if 'lyapunov' in rwd_fn_name:
                reward_item = scale * reward_fn(err_prev, err_next)
            elif 'fall_down' in rwd_fn_name:
                reward_item = scale * reward_fn(is_fail)
            elif 'jerky' in rwd_fn_name:
                reward_item = scale * reward_fn(delta_action)
            else:
                reward_item = scale * reward_fn()
            tot_rwd += reward_item
            self.episode_sums[rwd_fn_name] += reward_item

        return tot_rwd

    def fall_down_reward(self, is_fail):
        """Penalize when robot falls down"""
        fail_times = is_fail.sum()
        return -fail_times

    def lin_vel_z_reward(self):
        # Penalize z axis base linear velocity
        return torch.square(self._robot.base_velocity_body_frame[:, 2])

    def lin_vel_tracking_reward(self):
        err = self._env.torque_optimizer.tracking_error
        lin_vel_error = torch.sum(torch.square(err[:, 6:9]), dim=1)
        return torch.exp(-lin_vel_error / self._config.tracking_sigma)

    def ang_vel_tracking_reward(self):
        err = self._env.torque_optimizer.tracking_error
        ang_vel_error = torch.sum(torch.square(err[:, 9:12]), dim=1)
        return torch.exp(-ang_vel_error / self._config.tracking_sigma)

    def orientation_tracking_reward(self):
        err = self._env.torque_optimizer.tracking_error
        return -torch.sum(torch.square(err[:, 3:6]), dim=1)

    def height_tracking_reward(self):
        err = torch.abs(self._env.torque_optimizer.tracking_error)
        return -torch.sum(err[:, 2])

    def jerky_action_reward(self, delta_action: torch.Tensor):
        """Penalize jerky actions to get smooth control output"""
        return torch.sum(delta_action)

    def forward_speed_reward(self):
        return self._robot.base_velocity_body_frame[:, 0]

    def upright_reward(self):
        return self._robot.projected_gravity[:, 2]

    def alive_reward(self):
        """Reward for keeping alive"""
        return torch.ones(self._num_envs, device=self._device)

    def height_reward(self):
        err = self._env.torque_optimizer.tracking_error
        return -torch.square(err[:, 2])

    def foot_slipping_reward(self):
        foot_slipping = torch.sum(
            self._gait_generator.desired_contact_state * torch.sum(torch.square(
                self._robot.foot_velocities_in_world_frame[:, :, :2]),
                dim=2),
            dim=1) / 4
        foot_slipping = torch.clip(foot_slipping, 0, 1)
        # print(self._gait_generator.desired_contact_state)
        # print(self._robot.foot_velocities_in_world_frame)
        # print(foot_slipping)
        # input("Any Key...")
        return -foot_slipping

    def foot_clearance_reward(self, foot_height_thres=0.02):
        desired_contacts = self._gait_generator.desired_contact_state
        foot_height = self._robot.foot_height - 0.02  # Foot radius
        # print(f"Foot height: {foot_height}")
        foot_height = torch.clip(foot_height, 0,
                                 foot_height_thres) / foot_height_thres
        foot_clearance = torch.sum(
            torch.logical_not(desired_contacts) * foot_height, dim=1) / 4

        return foot_clearance

    def foot_force_reward(self):
        """Swing leg should not have contact force."""
        foot_forces = torch.norm(self._robot.foot_contact_forces, dim=2)
        calf_forces = torch.norm(self._robot.calf_contact_forces, dim=2)
        thigh_forces = torch.norm(self._robot.thigh_contact_forces, dim=2)
        limb_forces = (foot_forces + calf_forces + thigh_forces).clip(max=10)
        foot_mask = torch.logical_not(self._gait_generator.desired_contact_state)

        return -torch.sum(limb_forces * foot_mask, dim=1) / 4

    def cost_of_transport_reward(self):
        motor_power = torch.abs(0.3 * self._robot.motor_torques ** 2 +
                                self._robot.motor_torques *
                                self._robot.motor_velocities)
        commanded_vel = torch.sqrt(torch.sum(self._env.command[:, :2] ** 2, dim=1))
        return -torch.sum(motor_power, dim=1) / commanded_vel

    def energy_consumption_reward(self):
        """Reward for decreasing motor power"""
        return -self._env.motor_power

    def contact_consistency_reward(self):
        desired_contact = self._gait_generator.desired_contact_state
        actual_contact = torch.logical_or(self._robot.foot_contacts,
                                          self._robot.calf_contacts)
        actual_contact = torch.logical_or(actual_contact,
                                          self._robot.thigh_contacts)
        # print(f"Actual contact: {actual_contact}")
        return torch.sum(desired_contact == actual_contact, dim=1) / 4

    def com_distance_to_goal_squared_reward(self):
        base_position = self._robot.base_position_world
        # print("Base position: {}".format(base_position))
        # print("Desired landing: {}".format(self._env.desired_landing_position))
        # import pdb
        # pdb.set_trace()
        return -torch.sum(torch.square(
            (base_position[:, :2] - self._env.desired_landing_position[:, :2]) /
            (self._env._jumping_distance[:, 0:1])),
            dim=1)

    def swing_foot_vel_reward(self):
        foot_vel = torch.sum(self._robot.foot_velocities_in_base_frame ** 2, dim=2)
        contact_mask = torch.logical_not(
            self._env.gait_generator.desired_contact_state)
        return -torch.sum(foot_vel * contact_mask, dim=1) / (
                torch.sum(contact_mask, dim=1) + 0.001)

    def com_height_reward(self):
        # Helps robot jump higher over all
        return self._robot.base_position_world[:, 2].clip(max=0.5)

    def heading_reward(self):
        # print(self._robot.base_orientation_rpy[:, 2])
        # input("Any Key...")
        return -self._robot.base_orientation_rpy[:, 2] ** 2

    def out_of_bound_action_reward(self):
        exceeded_action = torch.maximum(
            self._env._action_lb - self._env._last_action,
            self._env._last_action - self._env._action_ub)
        exceeded_action = torch.clip(exceeded_action, min=0.)
        normalized_excess = exceeded_action / (self._env._action_ub -
                                               self._env._action_lb)
        return -torch.sum(torch.square(normalized_excess), dim=1)

    def swing_residual_reward(self):
        return -torch.mean(torch.square(self._env._last_action[:, -6:]), axis=1)

    def knee_contact_reward(self):
        rew = -((torch.sum(torch.logical_or(self._env.robot.thigh_contacts,
                                            self._env.robot.calf_contacts),
                           dim=1)).float()) / 4
        return rew

    def body_contact_reward(self):
        """Penalize for body contact"""
        return -self._robot.has_body_contact.float()

    def stepping_freq_reward(self):
        """Reward for jumping at low frequency."""
        return 1.5 - self._env.gait_generator.stepping_frequency.clip(min=1.5)

    def friction_cone_reward(self):
        return -self._env._num_clips / 4

    def lyapunov_reward(self, err, err_next):
        """Get lyapunov-like reward
            error: position_error     (p)
                   orientation_error  (rpy)
                   linear_vel_error   (v)
                   angular_vel_error  (w)
        """

        _MATRIX_P = torch.tensor(MATRIX_P, dtype=torch.float32, device=self._device)
        s_curr = err[:, 2:]
        s_next = err_next[:, 2:]
        # print(f"s: {s.shape}")
        # print(f"s_new: {s_new.shape}")
        # ly_reward_curr = s_new.T @ MATRIX_P @ s_new
        ST1 = torch.matmul(s_curr, _MATRIX_P)
        ly_reward_curr = torch.sum(ST1 * s_curr, dim=1, keepdim=True)

        # ly_reward_next = s_next_new.T @ MATRIX_P @ s_next_new
        ST2 = torch.matmul(s_next, _MATRIX_P)
        ly_reward_next = torch.sum(ST2 * s_next, dim=1, keepdim=True)

        sum_reward = ly_reward_curr - ly_reward_next  # multiply scaler to decrease
        # print(f"sum_reward: {sum_reward.shape}")
        # print(f"sum_reward: {sum_reward}")
        # sum_reward = torch.tensor(reward, device=self._device)

        return sum_reward.squeeze(dim=-1)

    def distance_to_wp_reward(self):
        """Get Waypoint Distance Reward
            curr_pos: 2d vector
            goal: 2d vector"""

        # distance = self._env.goal_distance
        # d_min = 0.0
        # d_max = 10.0
        # normalized_distance = (distance - d_min) / (d_max - d_min)

        # return -normalized_distance

        curr_distance = self._env.goal_distance
        last_distance = self._env.last_goal_distance

        alpha1 = 10
        step_rew = curr_distance - last_distance

        alpha2 = 0
        wp_prog_rew = 0
        for i in range(len(self._env.sub_goal_reach_time)):
            wp_prog_rew += self.wp_progress_reward(self._env.sub_goal_reach_time[i])

        return alpha1 * step_rew + alpha2 * wp_prog_rew

    def wp_progress_reward(self, step_cnt):
        """Reward for the reached waypoint progress"""
        time_decay_factor = 5e-2
        return math.exp(-time_decay_factor * step_cnt)

    def reach_wp_reward(self):
        """Reward for reaching the waypoint"""
        return int(self._env.sub_goal_reach_flag)

    def reach_time_reward(self):
        """Penalize robot travel time"""
        decay_factor = 1e-3
        return -torch.sum(self._env.episode_length_buf * decay_factor)

    def reach_goal_reward(self):
        """The reward for reaching the goal (destination)"""
        return int(not self._env.planner.planning_flag)
