import time

from quad_A1.a1_utilities.a1_sensor_histories import NormedStateHistory
from quad_A1.a1_utilities.a1_sensor_process import observation_to_joint_position, observation_to_torque
from quad_A1.a1_utilities.logger import StateLogger
from quad_A1.a1_utilities.a1_sensor_histories import VisualHistory
from quad_A1.a1_utilities.velocity_estimator import VelocityEstimator

import numpy as np


class PolicyWrapper():
    def __init__(
            self,
            robot,
            gym_params,
            policy,
            obs_normalizer_mean, obs_normalizer_var,
            save_dir_name,
            sliding_frames=True, no_tensor=False,
            default_joint_angle=None,
            action_range=None,
            state_only=False,
            clip_motor=False,
            clip_motor_value=0.5,
            use_foot_contact=False,
            save_log=False
    ):
        self.pf = policy
        self.no_tensor = no_tensor

        self.state_only = state_only

        if default_joint_angle == None:
            default_joint_angle = [0.0, 0.9, -1.8]
        self.default_joint_angle = np.array(default_joint_angle * 4)

        self.current_joint_angle = default_joint_angle
        self.current_velocity = [0.0, 0.0, 0.0]
        self.clip_motor = clip_motor
        self.clip_motor_value = clip_motor_value

        if action_range == None:
            action_range = [0.05, 0.5, 0.5]
        self.action_range = np.array(action_range * 4)

        self.action_lb = self.default_joint_angle - self.action_range
        self.action_ub = self.default_joint_angle + self.action_range

        self.velocity_estimator = VelocityEstimator(robot, gym_params)

        self.use_foot_contact = use_foot_contact
        last_start = 0
        if use_foot_contact:
            self.foot_contact_historical_data = NormedStateHistory(
                input_dim=4,
                num_hist=3,
                mean=obs_normalizer_mean[0:12],
                var=obs_normalizer_var[0:12]
            )
            last_start = 12

        self.imu_historical_data = NormedStateHistory(
            input_dim=4,
            num_hist=3,
            mean=obs_normalizer_mean[last_start: last_start + 12],
            var=obs_normalizer_var[last_start: last_start + 12]
        )

        self.joint_angle_historical_data = NormedStateHistory(
            input_dim=12,
            num_hist=3,
            mean=obs_normalizer_mean[last_start + 12: last_start + 48],
            var=obs_normalizer_var[last_start + 12: last_start + 48]
        )

        self.last_action_historical_data = NormedStateHistory(
            input_dim=12,
            num_hist=3,
            mean=obs_normalizer_mean[last_start + 48: last_start + 84],
            var=obs_normalizer_var[last_start + 48: last_start + 84]
        )

        self.velocity_historical_data = NormedStateHistory(
            input_dim=3,
            num_hist=3,
            mean=obs_normalizer_mean[last_start + 84: last_start + 93],
            var=obs_normalizer_var[last_start + 84: last_start + 93]
        )

    def process_obs(self, observation, last_action):
        # IMU
        imu_hist_normalized = self.imu_historical_data.record_and_normalize(
            np.array([
                observation.imu.rpy[0],
                observation.imu.rpy[1],
                observation.imu.gyroscope[0],
                observation.imu.gyroscope[1],
            ])
        )

        # joint angle
        joint_angle = observation_to_joint_position(observation)
        self.current_joint_angle = joint_angle
        joint_angle_hist_normalized = self.joint_angle_historical_data.record_and_normalize(
            joint_angle
        )

        # last action
        last_action_normalized = self.last_action_historical_data.record_and_normalize(last_action)

        # velocity
        self.velocity_estimator.update(time.time(), observation)
        velocity = self.velocity_estimator.estimated_velocity
        self.current_velocity = velocity
        velocity_hist_normalized = self.velocity_historical_data.record_and_normalize(velocity)

        # velocity
        obs_list = []

        if self.use_foot_contact:
            foot_contact_normalized = self.foot_contact_historical_data.record_and_normalize(
                np.array(observation.footForce) > 20)
            obs_list.append(foot_contact_normalized)

        obs_list += [
            velocity_hist_normalized,
            imu_hist_normalized,
            last_action_normalized,
            joint_angle_hist_normalized,
        ]

        obs_normalized_np = np.hstack(obs_list)

        if not self.no_tensor:
            import torch
            ob_t = torch.Tensor(obs_normalized_np).unsqueeze(0).to("cuda:0")
        else:
            ob_t = obs_normalized_np[np.newaxis, :]
        return ob_t

    def process_act(self, action):
        if self.clip_motor:
            action = np.clip(
                action,
                self.current_joint_angle - self.clip_motor_value,
                self.current_joint_angle + self.clip_motor_value
            )
        return action

    def get_action(self, observation, last_action):
        """
        This function process raw observation, fed normalized observation into
        the network, de-normalize and output the action.
        """
        ob_t = self.process_obs(observation, last_action).reshape(-1)
        action = self.pf.get_actions(ob_t, mode='eval')
        action = self.process_act(action)
        return action
