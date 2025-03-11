"""Simple state estimator for Go2 robot."""
import time
from typing import Sequence

import numpy as np

import pybullet as p
from filterpy.kalman import KalmanFilter

from src.envs.robots.modules.utils.moving_window_filter import MovingWindowFilter

_DEFAULT_WINDOW_SIZE = 1
_ANGULAR_VELOCITY_FILTER_WINDOW_SIZE = 1
_GROUND_NORMAL_WINDOW_SIZE = 20


def convert_to_skew_symmetric(x: np.ndarray) -> np.ndarray:
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


DTYPE = np.float32


class EstimatedStates:
    def __init__(self):
        self.position = np.zeros((3, 1), dtype=DTYPE)
        self.vWorld = np.zeros((3, 1), dtype=DTYPE)
        self.omegaWorld = np.zeros((3, 1), dtype=DTYPE)
        self.orientation = Quaternion(1, 0, 0, 0)

        self.rBody = np.zeros((3, 3), dtype=DTYPE)
        self.rpy = np.zeros((3, 1), dtype=DTYPE)
        self.rpyBody = np.zeros((3, 1), dtype=DTYPE)

        self.ground_normal_world = np.array([0, 0, 1], dtype=DTYPE)
        self.ground_normal_yaw = np.array([0, 0, 1], dtype=DTYPE)

        self.vBody = np.zeros((3, 1), dtype=DTYPE)
        self.omegaBody = np.zeros((3, 1), dtype=DTYPE)


class StateEstimator:
    """Estimates base velocity of quadrupedal robot.
    The velocity estimator consists of a state estimator for CoM velocity.
    Two sources of information are used:
    The integrated reading of accelerometer and the velocity estimation from
    contact legs. The readings are fused together using a Kalman Filter.
    """

    def __init__(self,
                 robot,
                 accelerometer_variance: np.ndarray = np.array(
                     [1.42072319e-05, 1.57958752e-05, 8.75317619e-05, 2e-5]),
                 sensor_variance: np.ndarray = np.array(
                     [0.33705298, 0.14858707, 0.68439632, 0.68]) * 0.03,
                 initial_variance: float = 0.1,
                 use_external_contact_estimator: bool = False):
        """Initiates the velocity/height estimator.
        See filterpy documentation in the link below for more details.
        https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html
        Args:
          robot: the robot class for velocity estimation.
          accelerometer_variance: noise estimation for accelerometer reading.
          sensor_variance: noise estimation for motor velocity reading.
          initial_variance: covariance estimation of initial state.
        """
        self.robot = robot
        self._envs = robot.num_envs
        self._use_external_contact_estimator = use_external_contact_estimator
        self._foot_contact = np.ones(4)

        self.filter = KalmanFilter(dim_x=4, dim_z=4, dim_u=4)
        self.filter.x = np.array([0., 0., 0., 0.26])
        self._initial_variance = initial_variance
        self._accelerometer_variance = accelerometer_variance
        self._sensor_variance = sensor_variance
        self.filter.P = np.eye(4) * self._initial_variance  # State covariance
        self.filter.Q = np.eye(4) * accelerometer_variance
        self.filter.R = np.eye(4) * sensor_variance

        self.filter.H = np.eye(4)  # measurement function (y=H*x)
        self.filter.F = np.eye(4)  # state transition matrix
        self.filter.B = np.eye(4)

        self.ma_filter = MovingWindowFilter(window_size=_DEFAULT_WINDOW_SIZE)
        self._angular_velocity_filter = MovingWindowFilter(window_size=_ANGULAR_VELOCITY_FILTER_WINDOW_SIZE)
        self._angular_velocity = np.zeros(3)
        self._estimated_velocity = np.zeros(3)
        self._estimated_position = np.array([0., 0., self.robot.mpc_body_height])

        # Estimate Ground Normal (in body frame)
        self._ground_normal = np.array([0., 0., 1.])
        self._ground_normal_window_size = _GROUND_NORMAL_WINDOW_SIZE
        self._ground_normal_filter = MovingWindowFilter(window_size=self._ground_normal_window_size)

        self.reset()

    def reset(self):
        self.filter.x = np.array([0., 0., 0., 0.26])
        self.filter.P = np.eye(4) * self._initial_variance
        self._last_timestamp = 0
        self._estimated_velocity = self.filter.x.copy()[:3]

    def _compute_delta_time(self, robot_state):
        del robot_state  # unused
        if self._last_timestamp == 0.:
            # First timestamp received, return an estimated delta_time.
            delta_time_s = self.robot.control_timestep
        else:
            delta_time_s = self.robot.time_since_reset_scalar - self._last_timestamp
        self._last_timestamp = self.robot.time_since_reset_scalar
        return delta_time_s

    def _get_velocity_and_height_observation(self):
        foot_positions = self.robot.foot_center_positions_in_base_frame_numpy
        rot_mat = self.robot.base_rot_mat_numpy
        ang_vel_cross = convert_to_skew_symmetric(self._angular_velocity)
        observed_velocities, observed_heights = [], []
        if self._use_external_contact_estimator:
            foot_contact = self._foot_contact.copy()
        else:
            foot_contact = self.robot.foot_contact_numpy
        for leg_id in range(4):
            if foot_contact[leg_id]:
                jacobian = self.robot.compute_foot_jacobian_tip(leg_id)
                # Only pick the jacobian related to joint motors
                joint_velocities = self.robot.motor_velocities_numpy[leg_id * 3:(leg_id + 1) * 3]
                leg_velocity_in_base_frame = jacobian.dot(joint_velocities)[:3]
                observed_velocities.append(
                    -rot_mat.dot(leg_velocity_in_base_frame +
                                 ang_vel_cross.dot(foot_positions[leg_id])))
                observed_heights.append(-rot_mat.dot(foot_positions[leg_id])[2] + 0.02)

        return observed_velocities, observed_heights

    def update_foot_contact(self, foot_contact):
        self._foot_contact = foot_contact[0, :].cpu().numpy().reshape(4)

    def update(self, robot_state):
        """Propagate current state estimate with new accelerometer reading."""

        delta_time_s = self._compute_delta_time(robot_state)
        sensor_acc = np.array(robot_state.imu.accelerometer)
        rot_mat = self.robot.base_rot_mat_numpy
        calibrated_acc = np.zeros(4)
        calibrated_acc[:3] = rot_mat.dot(sensor_acc) + np.array([0., 0., -9.8])
        calibrated_acc[3] = self._estimated_velocity[2]
        self.filter.predict(u=calibrated_acc * delta_time_s)

        (observed_velocities,
         observed_heights) = self._get_velocity_and_height_observation()

        if observed_velocities:
            observed_velocities = np.mean(observed_velocities, axis=0)
            observed_heights = np.mean(observed_heights)
            self.filter.update(
                np.concatenate((observed_velocities, [observed_heights])))

        self._estimated_velocity = self.ma_filter.calculate_average(
            self.filter.x.copy()[:3])
        self._angular_velocity = self._angular_velocity_filter.calculate_average(
            np.array(robot_state.imu.gyroscope))

        self._estimated_position += delta_time_s * self._estimated_velocity
        self._estimated_position[2] = self.filter.x.copy()[3]

    def update_sim(self, robot_state):
        """Propagate current state estimate with new accelerometer reading."""

        delta_time_s = self._compute_delta_time(robot_state)
        sensor_acc = np.array(robot_state.imu.accelerometer)
        rot_mat = self.robot.base_rot_mat
        calibrated_acc = np.zeros(4)
        calibrated_acc[:3] = rot_mat.dot(sensor_acc) + np.array([0., 0., -9.8])
        calibrated_acc[3] = self._estimated_velocity[2]
        self.filter.predict(u=calibrated_acc * delta_time_s)

        (observed_velocities,
         observed_heights) = self._get_velocity_and_height_observation()

        if observed_velocities:
            observed_velocities = np.mean(observed_velocities, axis=0)
            observed_heights = np.mean(observed_heights)
            self.filter.update(np.concatenate((observed_velocities, [observed_heights])))

        self._estimated_velocity = self.ma_filter.calculate_average(
            self.filter.x.copy()[:3])
        self._angular_velocity = self._angular_velocity_filter.calculate_average(
            np.array(self.robot.base_angular_velocity_body_frame))

        self._estimated_position += delta_time_s * self._estimated_velocity
        self._estimated_position[2] = self.filter.x.copy()[3]

    def update_ground_normal_vec(self):
        # Compute ground normal
        _ground_normal_vec = self._compute_ground_normal(self.robot.foot_position_history[0, :])

        # Obtain smoothed ground normal vector
        self._ground_normal = self._ground_normal_filter.calculate_average(_ground_normal_vec)
        self._ground_normal /= np.linalg.norm(self._ground_normal)

        print(f"_ground_normal: {self._ground_normal}")
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        # time.sleep(0.5)

    # def _update_com_position_ground_frame(self, foot_positions: np.ndarray):
    #     foot_contacts = self._contactPhase.flatten()
    #     if np.sum(foot_contacts) == 0:
    #         return np.array((0, 0, self.body_height))
    #     else:
    #         foot_positions_ground_frame = (foot_positions.reshape((4, 3)).dot(self.ground_R_body_frame.T))
    #         foot_heights = -foot_positions_ground_frame[:, 2]
    #
    #     height_in_ground_frame = np.sum(foot_heights * foot_contacts) / np.sum(foot_contacts)
    #     self.result.position[2] = height_in_ground_frame

    def _compute_ground_normal(self, foot_positions_body_frame: np.ndarray):
        """
        Computes the surface orientation in robot frame based on foot positions.
        Solves a least-squares problem, see the following paper for details:
        https://ieeexplore.ieee.org/document/7354099
        """
        print(f"foot_positions_body_frame: {foot_positions_body_frame}")
        contact_foot_positions = np.asarray(foot_positions_body_frame)
        normal_vec = np.linalg.lstsq(contact_foot_positions, np.ones(4))[0]
        normal_vec /= np.linalg.norm(normal_vec)
        if normal_vec[2] < 0:
            normal_vec = -normal_vec
        return normal_vec

    @property
    def estimated_velocity(self):
        return self._estimated_velocity.copy()

    @property
    def estimated_position(self):
        return self._estimated_position.copy()

    @property
    def angular_velocity(self):
        return self._angular_velocity

    @property
    def use_external_contact_estimator(self):
        return self._use_external_contact_estimator

    @use_external_contact_estimator.setter
    def use_external_contact_estimator(self, use_external_contact_estimator):
        self._use_external_contact_estimator = use_external_contact_estimator

    @property
    def ground_normal(self):
        return self._ground_normal

    @property
    def gravity_projection_vector(self):
        _, world_orientation_ground_frame = p.invertTransform(
            [0., 0., 0.], self.ground_orientation_in_world_frame)
        return np.array(
            p.multiplyTransforms([0., 0., 0.], world_orientation_ground_frame,
                                 [0., 0., 1.], [0., 0., 0., 1.])[0])

    @property
    def ground_orientation_in_robot_frame(self):
        normal_vec = self.ground_normal
        axis = np.array([-normal_vec[1], normal_vec[0], 0])
        axis /= np.linalg.norm(axis)
        angle = np.arccos(normal_vec[2])
        return np.array(p.getQuaternionFromAxisAngle(axis, angle))

    @property
    def ground_orientation_in_world_frame(self) -> Sequence[float]:
        return np.array(
            p.multiplyTransforms([0., 0., 0.], self.robot.base_orientation_quat[0, :],
                                 [0., 0., 0.],
                                 self.ground_orientation_in_robot_frame)[1])

    @property
    def com_orientation_quaternion_in_ground_frame(self):
        _, orientation = p.invertTransform([0., 0., 0.], self.ground_orientation_in_robot_frame)
        return np.array(orientation)

    @property
    def com_position_in_ground_frame(self):
        foot_contacts = self.robot.foot_contacts[0, :].cpu().numpy()

        if np.sum(foot_contacts) == 0:  # No feet on the ground
            return np.array((0, 0, self.robot.mpc_body_height))
        else:
            foot_positions_robot_frame = self.robot.foot_positions_in_base_frame[0, :].cpu().numpy()

            ground_orientation_matrix_robot_frame = p.getMatrixFromQuaternion(
                self.ground_orientation_in_robot_frame)
            print(f"self._ground_norm_vec: {self._ground_normal}")
            print(f"ground_orientation_matrix_robot_frame: {ground_orientation_matrix_robot_frame}")
            # time.sleep(0.1)

            # Reshape
            ground_orientation_matrix_robot_frame = np.array(
                ground_orientation_matrix_robot_frame).reshape((3, 3))

            foot_positions_ground_frame = (foot_positions_robot_frame.dot(
                ground_orientation_matrix_robot_frame.T))

            foot_heights = -foot_positions_ground_frame[:, 2]

            # print(f"foot_positions_robot_frame: {foot_positions_robot_frame}")
            # print(f"foot_positions_ground_frame: {foot_positions_ground_frame}")

            return np.array((
                0,
                0,
                np.sum(foot_heights * foot_contacts) / np.sum(foot_contacts),
            ))

    def estimate_slope_sim(self):
        from isaacgym import gymapi
        import torch
        orientation = self.robot.base_orientation_quat[0]
        print(f"ori: {orientation}")
        # 计算地面法向量
        quat = gymapi.Quat(orientation[0].item(), orientation[1].item(), orientation[2].item(),
                           orientation[3].item())
        up_vector = quat.rotate(gymapi.Vec3(0, 0, 1))  # 世界 z 轴方向

        # 计算坡度角度（与 z 轴夹角）
        slope_angle = torch.acos(torch.tensor(up_vector.z))  # z 分量
        slope_deg = slope_angle * 180.0 / 3.14159  # 转换为角度
        print(f"slope_deg: {slope_deg}")
