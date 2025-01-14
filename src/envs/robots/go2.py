"""Vectorized Go2 robot in Isaac Gym."""
from typing import Any, Sequence

from isaacgym.torch_utils import to_torch
import ml_collections
import torch

from src.envs.robots.robot import Robot
from src.envs.robots.motors import MotorControlMode, MotorGroup, MotorModel

_ARRAY = Sequence[float]


@torch.jit.script
def motor_angles_from_foot_positions(foot_local_positions,
                                     hip_offset,
                                     device: str = "cuda"):
    foot_positions_in_hip_frame = foot_local_positions - hip_offset
    l_up = 0.213
    l_low = 0.233
    l_hip = 0.08 * torch.tensor([-1, 1, -1, 1], device=device)

    x = foot_positions_in_hip_frame[:, :, 0]
    y = foot_positions_in_hip_frame[:, :, 1]
    z = foot_positions_in_hip_frame[:, :, 2]
    theta_knee = -torch.arccos(
        torch.clip((x ** 2 + y ** 2 + z ** 2 - l_hip ** 2 - l_low ** 2 - l_up ** 2) /
                   (2 * l_low * l_up), -1, 1))
    l = torch.sqrt(
        torch.clip(l_up ** 2 + l_low ** 2 + 2 * l_up * l_low * torch.cos(theta_knee),
                   1e-7, 1))
    theta_hip = torch.arcsin(torch.clip(-x / l, -1, 1)) - theta_knee / 2
    c1 = l_hip * y - l * torch.cos(theta_hip + theta_knee / 2) * z
    s1 = l * torch.cos(theta_hip + theta_knee / 2) * y + l_hip * z
    theta_ab = torch.arctan2(s1, c1)

    # thetas: num_envs x 4
    joint_angles = torch.stack([theta_ab, theta_hip, theta_knee], dim=2)
    return joint_angles.reshape((-1, 12))


class Go2(Robot):
    """Go2 robot in simulation."""

    def __init__(
            self,
            sim: Any,
            viewer: Any,
            sim_config: ml_collections.ConfigDict(),
            num_envs: int,
            world_env: Any,
            init_positions: torch.Tensor,
            motor_control_mode: MotorControlMode,
            motor_torque_delay_steps: int = 0,
    ):
        motors = MotorGroup(device=sim_config.sim_device,
                            num_envs=num_envs,
                            motors=(
                                MotorModel(
                                    name="FR_hip_joint",
                                    motor_control_mode=motor_control_mode,
                                    init_position=0.0,
                                    min_position=-0.802851455917,
                                    max_position=0.802851455917,
                                    min_velocity=-30,
                                    max_velocity=30,
                                    min_torque=-23.7,
                                    max_torque=23.7,
                                    kp=100,
                                    kd=1,
                                ),
                                MotorModel(
                                    name="FR_thigh_joint",
                                    motor_control_mode=motor_control_mode,
                                    init_position=0.9,
                                    min_position=-1.0471975512,
                                    max_position=4.18879020479,
                                    min_velocity=-30,
                                    max_velocity=30,
                                    min_torque=-23.7,
                                    max_torque=23.7,
                                    kp=100,
                                    kd=1,
                                ),
                                MotorModel(
                                    name="FR_calf_joint",
                                    motor_control_mode=motor_control_mode,
                                    init_position=-1.8,
                                    min_position=-2.6965336943,
                                    max_position=-0.916297857297,
                                    min_velocity=-20,
                                    max_velocity=20,
                                    min_torque=-35.55,
                                    max_torque=35.55,
                                    kp=100,
                                    kd=1,
                                ),
                                MotorModel(
                                    name="FL_hip_joint",
                                    motor_control_mode=motor_control_mode,
                                    init_position=0.0,
                                    min_position=-0.802851455917,
                                    max_position=0.802851455917,
                                    min_velocity=-30,
                                    max_velocity=30,
                                    min_torque=-23.7,
                                    max_torque=23.7,
                                    kp=100,
                                    kd=1,
                                ),
                                MotorModel(
                                    name="FL_thigh_joint",
                                    motor_control_mode=motor_control_mode,
                                    init_position=0.9,
                                    min_position=-1.0471975512,
                                    max_position=4.18879020479,
                                    min_velocity=-30,
                                    max_velocity=30,
                                    min_torque=-23.7,
                                    max_torque=23.7,
                                    kp=100,
                                    kd=1,
                                ),
                                MotorModel(
                                    name="FL_calf_joint",
                                    motor_control_mode=motor_control_mode,
                                    init_position=-1.8,
                                    min_position=-1.0471975512,
                                    max_position=4.18879020479,
                                    min_velocity=-20,
                                    max_velocity=20,
                                    min_torque=-35.55,
                                    max_torque=35.55,
                                    kp=100,
                                    kd=1,
                                ),
                                MotorModel(
                                    name="RR_hip_joint",
                                    motor_control_mode=motor_control_mode,
                                    init_position=0.0,
                                    min_position=-0.802851455917,
                                    max_position=0.802851455917,
                                    min_velocity=-30,
                                    max_velocity=30,
                                    min_torque=-23.7,
                                    max_torque=23.7,
                                    kp=100,
                                    kd=1,
                                ),
                                MotorModel(
                                    name="RR_thigh_joint",
                                    motor_control_mode=motor_control_mode,
                                    init_position=0.9,
                                    min_position=-1.0471975512,
                                    max_position=4.18879020479,
                                    min_velocity=-30,
                                    max_velocity=30,
                                    min_torque=-23.7,
                                    max_torque=23.7,
                                    kp=100,
                                    kd=1,
                                ),
                                MotorModel(
                                    name="RR_calf_joint",
                                    motor_control_mode=motor_control_mode,
                                    init_position=-1.8,
                                    min_position=-2.6965336943,
                                    max_position=-0.916297857297,
                                    min_velocity=-20,
                                    max_velocity=20,
                                    min_torque=-35.55,
                                    max_torque=35.55,
                                    kp=100,
                                    kd=1,
                                ),
                                MotorModel(
                                    name="RL_hip_joint",
                                    motor_control_mode=motor_control_mode,
                                    init_position=0.0,
                                    min_position=-0.802851455917,
                                    max_position=0.802851455917,
                                    min_velocity=-30,
                                    max_velocity=30,
                                    min_torque=-23.7,
                                    max_torque=23.7,
                                    kp=100,
                                    kd=1,
                                ),
                                MotorModel(
                                    name="RL_thigh_joint",
                                    motor_control_mode=motor_control_mode,
                                    init_position=0.9,
                                    min_position=-1.0471975512,
                                    max_position=4.18879020479,
                                    min_velocity=-30,
                                    max_velocity=30,
                                    min_torque=-23.7,
                                    max_torque=23.7,
                                    kp=100,
                                    kd=1,
                                ),
                                MotorModel(
                                    name="RL_calf_joint",
                                    motor_control_mode=motor_control_mode,
                                    init_position=-1.8,
                                    min_position=-2.6965336943,
                                    max_position=-0.916297857297,
                                    min_velocity=-20,
                                    max_velocity=20,
                                    min_torque=-35.55,
                                    max_torque=35.55,
                                    kp=100,
                                    kd=1,
                                ),
                            ),
                            torque_delay_steps=motor_torque_delay_steps)

        com_offset = -to_torch([0.011611, 0.004437, 0.000108],
                               device=sim_config.sim_device)
        self._hip_offset = to_torch(
            [[0.1881, -0.04675, 0.], [0.1881, 0.04675, 0.], [-0.1881, -0.04675, 0.],
             [-0.1881, 0.04675, 0.]],
            device=sim_config.sim_device) + com_offset

        delta_x, delta_y = 0.0, 0.0
        hip_position_single = to_torch((
            (0.1835 + delta_x, -0.131 - delta_y, 0),
            (0.1835 + delta_x, 0.122 + delta_y, 0),
            (-0.1926 - delta_x, -0.131 - delta_y, 0),
            (-0.1926 - delta_x, 0.122 + delta_y, 0),
        ), device=sim_config.sim_device)
        self._hip_positions_in_body_frame = torch.stack([hip_position_single] *
                                                        num_envs,
                                                        dim=0)
        super().__init__(sim=sim,
                         viewer=viewer,
                         num_envs=num_envs,
                         world_env=world_env,
                         init_positions=init_positions,
                         urdf_path="resources/go2/urdf/go2_ordered.urdf",
                         sim_config=sim_config,
                         motors=motors,
                         feet_names=[
                             "1_FR_foot",
                             "2_FL_foot",
                             "3_RR_foot",
                             "4_RL_foot",
                         ],
                         calf_names=[
                             "1_FR_calf",
                             "2_FL_calf",
                             "3_RR_calf",
                             "4_RL_calf",
                         ],
                         thigh_names=[
                             "1_FR_thigh",
                             "2_FL_thigh",
                             "3_RR_thigh",
                             "4_RL_thigh",
                         ])
        # from src.envs.robots.modules.estimator import state_estimator
        # self._state_estimator = state_estimator.StateEstimator(self, use_external_contact_estimator=False)

    # def step(self, action):
    #     from isaacgym import gymtorch
    #     for _ in range(self._sim_config.action_repeat):
    #         self._torques, _ = self.motor_group.convert_to_torque(
    #             action, self._motor_positions, self._motor_velocities)
    #         # time.sleep(1)
    #         self._gym.set_dof_actuation_force_tensor(
    #             self._sim, gymtorch.unwrap_tensor(self._torques))
    #         self._gym.simulate(self._sim)
    #         # if self._device == "cpu":
    #         self._gym.fetch_results(self._sim, True)
    #
    #         self._gym.refresh_dof_state_tensor(self._sim)
    #         self._time_since_reset += self._sim_config.sim_params.dt
    #
    #         # self._state_estimator.update_sim(self._raw_state)
    #         # self._state_estimator.update_ground_normal_vec()
    #         # self._state_estimator.update_foot_contact()
    #
    #     self._post_physics_step()
    #
    # def reset_idx(self, env_ids):
    #     super().reset_idx(env_ids)
    #     self._state_estimator.reset()
    #
    # @property
    # def base_position(self):
    #     return to_torch([self._state_estimator.estimated_position], device=self._device)
    #
    # @property
    # def base_velocity_world_frame(self):
    #     return to_torch(self._state_estimator.estimated_velocity[None, :], device=self._device)
    #
    # @property
    # def base_velocity_body_frame(self):
    #     return to_torch(self._base_rot_mat.T.dot(self._state_estimator.estimated_velocity)[None, :],
    #                     device=self._device)
    #
    # @property
    # def base_angular_velocity_body_frame(self):
    #     """Smoothed using moving-window filter"""
    #     return to_torch(self._state_estimator.angular_velocity[None, :], device=self._device)
    #
    # @property
    # def base_angular_velocity_world_frame(self):
    #     return to_torch(self._base_rot_mat.T.dot(self._state_estimator.angular_velocity)[None, :],
    #                     device=self._device)
    #
    # @property
    # def state_estimator(self):
    #     return self._state_estimator

    @property
    def hip_positions_in_body_frame(self):
        return self._hip_positions_in_body_frame

    @property
    def hip_offset(self):
        """Position of hip offset in base frame, used for IK only."""
        return self._hip_offset

    def get_motor_angles_from_foot_positions(self, foot_local_positions):
        return motor_angles_from_foot_positions(foot_local_positions,
                                                self.hip_offset,
                                                device=self._device)
