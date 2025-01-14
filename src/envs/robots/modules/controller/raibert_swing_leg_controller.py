"""Implements the Raibert Swing Leg controller in Isaac."""
import time
from typing import Any, Tuple

import numpy as np
import torch

_LANDING_CLIP_X = (-0.15, 0.15)
_LANDING_CLIP_Y = (-0.08, 0.08)


@torch.jit.script
def cubic_bezier(x0: torch.Tensor,
                 x1: torch.Tensor,
                 t: torch.Tensor) -> torch.Tensor:
    progress = t ** 3 + 3 * t ** 2 * (1 - t)
    return x0 + progress * (x1 - x0)


@torch.jit.script
def _gen_swing_foot_trajectory(input_phase: torch.Tensor,
                               start_pos: torch.Tensor,
                               mid_pos: torch.Tensor,
                               end_pos: torch.Tensor) -> torch.Tensor:
    cutoff = 0.5  # time point phase for cutoff
    input_phase = torch.stack([input_phase] * 3, dim=-1)
    return torch.where(input_phase < cutoff,
                       cubic_bezier(start_pos, mid_pos, input_phase / cutoff),
                       cubic_bezier(mid_pos, end_pos, (input_phase - cutoff) / (1 - cutoff)))


@torch.jit.script
def cross_quad(v1, v2):
    """Assumes v1 is nx3, v2 is nx4x3"""
    v1 = torch.stack([v1, v1, v1, v1], dim=1)
    shape = v1.shape
    v1 = v1.reshape((-1, 3))
    v2 = v2.reshape((-1, 3))
    return torch.cross(v1, v2).reshape((shape[0], shape[1], 3))


# @torch.jit.script
def compute_desired_foot_positions(
        robot,
        base_rot_mat,
        base_height,
        hip_positions_in_body_frame,
        base_velocity_world_frame,
        base_angular_velocity_body_frame,
        projected_gravity,
        desired_base_height: float,
        foot_height: float,
        foot_landing_clearance: float,
        stance_duration,
        normalized_phase,
        phase_switch_foot_positions,
        clip_x: Tuple[float, float] = _LANDING_CLIP_X,
        clip_y: Tuple[float, float] = _LANDING_CLIP_Y,
):
    """
    The foot position is calculated under the assumption that each leg is in swing mode.
    This will later be masked when generating motor actions according to their actual gait status
    """
    hip_position = torch.matmul(base_rot_mat,
                                hip_positions_in_body_frame.transpose(
                                    1, 2)).transpose(1, 2)
    # Mid-air position
    mid_position = torch.clone(hip_position)
    mid_position[..., 2] = (-base_height[:, None] + foot_height)
    print("____________________________________________________________")
    print(f"hip_position: {hip_position}")
    print(f"mid_position: {mid_position}")
    print(f"base_height: {base_height}")
    print(f"foot_height: {foot_height}")
    print("____________________________________________________________")
    # / projected_gravity[:, 2])[:, None]

    # Land position
    base_velocity = base_velocity_world_frame
    hip_velocity_body_frame = cross_quad(base_angular_velocity_body_frame,
                                         hip_positions_in_body_frame)
    hip_velocity = base_velocity[:, None, :] + torch.matmul(
        base_rot_mat, hip_velocity_body_frame.transpose(1, 2)).transpose(1, 2)

    # delta_x = v_x * T_stance / 2
    land_position = hip_velocity * stance_duration[:, :, None] / 2

    # Clip Foot landing position in xy-plane
    land_position[..., 0] = torch.clip(land_position[..., 0], clip_x[0], clip_x[1])
    land_position[..., 1] = torch.clip(land_position[..., 1], clip_y[0], clip_y[1])

    # Foot landing position in world frame (p_land = hip_pos_world_frame + v_com * T_stance / 2)
    land_position += hip_position

    land_position[..., 2] = (-base_height[:, None] + foot_landing_clearance)
    # -land_position[..., 0] * projected_gravity[:, 0, None]
    # -land_position[..., 1] * projected_gravity[:, 1, None]
    # ) / projected_gravity[:, 2, None]

    # Compute target position compensation due to slope
    # gravity_projection_vector = robot.state_estimator.gravity_projection_vector
    #
    # desired_landing_height = base_height[:, None] + foot_landing_clearance
    # desired_landing_height = np.asarray(desired_landing_height)
    #
    # multiplier = -desired_landing_height[0] / gravity_projection_vector[2]
    #
    # land_position[0, ..., :2] += gravity_projection_vector[:2] * multiplier
    # print(f"gravity_projection_vector: {gravity_projection_vector}")
    # print(f"desired_landing_height: {desired_landing_height}")
    # print(f"multiplier: {multiplier}")
    # print(f"land_position: {land_position}")
    # print(f"land_position[0, ..., :2]: +{land_position[0, ..., :2]}")

    # time.sleep(123)
    # print(f"normalized_phase: {normalized_phase}")
    # print(f"phase_switch_foot_positions: {phase_switch_foot_positions}")
    # print(f"mid_position: {mid_position}")

    foot_position = _gen_swing_foot_trajectory(input_phase=normalized_phase,
                                               start_pos=phase_switch_foot_positions,
                                               mid_pos=mid_position,
                                               end_pos=land_position)
    # print(f"Land position: {land_position}")
    # print(f"Foot position: {foot_position}")
    # time.sleep(123)
    # ans = input("Any Key...")
    # if ans in ["Y", "y"]:
    #   import pdb
    #   pdb.set_trace()

    return foot_position


class RaibertSwingLegController:
    """Controls the swing leg position using Raibert's formula.
    For details, please refer to chapter 2 in "Legged robots that balance" by
    Marc Raibert. The key idea is to stabilize the swing foot's location based on
    the CoM moving speed.
    """

    def __init__(self,
                 robot: Any,
                 gait_generator: Any,
                 desired_base_height: float = 0.3,
                 foot_landing_clearance: float = 0.01,
                 foot_height: float = 0.15):
        self._robot = robot
        self._device = robot.device
        self._num_envs = robot.num_envs
        self._gait_generator = gait_generator
        self._last_leg_state = gait_generator.desired_contact_state
        self._foot_landing_clearance = foot_landing_clearance
        self._desired_base_height = desired_base_height
        self._foot_height = foot_height
        self._phase_switch_foot_positions = None
        self.reset()

    def reset(self) -> None:
        self._last_leg_state = torch.clone(self._gait_generator.desired_contact_state)
        self._phase_switch_foot_positions = torch.matmul(
            self._robot.base_rot_mat,
            self._robot.foot_positions_in_base_frame.transpose(1, 2)).transpose(1, 2)

    def reset_idx(self, env_ids) -> None:
        self._last_leg_state[env_ids] = torch.clone(self._gait_generator.desired_contact_state[env_ids])
        self._phase_switch_foot_positions[env_ids] = torch.matmul(
            self._robot.base_rot_mat[env_ids],
            self._robot.foot_positions_in_base_frame[env_ids].transpose(1, 2)).transpose(1, 2)

    def update(self) -> None:
        new_leg_state = torch.clone(self._gait_generator.desired_contact_state)
        new_foot_positions = torch.matmul(
            self._robot.base_rot_mat,
            self._robot.foot_positions_in_base_frame.transpose(1, 2)).transpose(1, 2)
        self._phase_switch_foot_positions = torch.where(
            torch.tile((self._last_leg_state == new_leg_state)[:, :, None], [1, 1, 3]),
            self._phase_switch_foot_positions, new_foot_positions)
        self._last_leg_state = new_leg_state

    @property
    def desired_foot_positions(self):
        """Computes desired foot positions in WORLD frame centered at robot base.

        Note: it returns an invalid position for stance legs.
        """
        desired_foot_pos = compute_desired_foot_positions(
            self._robot,
            self._robot.base_rot_mat,
            self._robot.base_position[:, 2],
            self._robot.hip_positions_in_body_frame,
            self._robot.base_velocity_world_frame,
            self._robot.base_angular_velocity_body_frame,
            self._robot.projected_gravity,
            self._desired_base_height,
            self._foot_height,
            self._foot_landing_clearance,
            self._gait_generator.stance_duration,
            self._gait_generator.normalized_phase,
            self._phase_switch_foot_positions,
        )
        # print(f"desired_foot_pos: {desired_foot_pos}")

        return desired_foot_pos
