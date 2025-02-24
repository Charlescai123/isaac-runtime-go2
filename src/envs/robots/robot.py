"""Abstract class for (vectorized) robots."""
import os
import sys
import time
from typing import Any, List

from isaacgym import gymtorch
from isaacgym.torch_utils import to_torch
import ml_collections
import torch

from src.configs.defaults import asset_options as asset_options_config
from src.envs.robots.utils.rotation_utils import quat_to_rot_mat, get_euler_zyx_from_quaternion
from src.envs.robots.modules.sensor.rgbd_camera import RGBDCamera
from isaacgym.terrain_utils import *


def angle_normalize(x):
    return torch.remainder(x + torch.pi, 2 * torch.pi) - torch.pi


class Robot:
    """General class for simulated quadrupedal robot."""

    def __init__(
            self,
            sim: Any,
            viewer: Any,
            world_env: Any,
            num_envs: int,
            init_positions: torch.Tensor,
            urdf_path: str,
            sim_config: ml_collections.ConfigDict,
            motors: Any,
            feet_names: List[str],
            calf_names: List[str],
            thigh_names: List[str],
    ):
        """Initializes the robot class."""
        self._sim = sim
        self._gym = gymapi.acquire_gym()
        self._viewer = viewer
        self._enable_viewer_sync = True
        self._sim_config = sim_config
        self._device = self._sim_config.sim_device
        self._num_envs = num_envs
        self._world_env_type = world_env

        self._motors = motors
        self._feet_names = feet_names
        self._calf_names = calf_names
        self._thigh_names = thigh_names

        self._scene_offset_x = 40  # start from a brighter place in chess plane
        init_positions[:, 0] = init_positions[:, 0] + self._scene_offset_x + 6  # (For testing)

        self._init_mpc_height = init_positions[:, 2]
        self._base_init_state = self._compute_base_init_state(init_positions)
        self._init_motor_angles = self._motors.init_positions
        self._envs = []
        self._world_envs = []
        self._robot_actors = []
        self._camera_sensors = []

        self._robot_actors_global_indices = []
        self._robot_rigid_body_global_indices = []

        self.record_video = False  # Record a video or not

        if "cuda" in self._device:
            torch._C._jit_set_profiling_mode(False)
            torch._C._jit_set_profiling_executor(False)

        self._load_urdf(urdf_path)
        self._gym.prepare_sim(self._sim)

        self._frames = []

        self._init_buffers()

        self._last_timestamp = torch.zeros(self._num_envs, device=self._device)
        self._time_since_reset = torch.zeros(self._num_envs, device=self._device)

        self._foot_positions_prev = self.foot_positions_in_base_frame.clone()
        self._foot_positions_prev[:, :, 2] = -self._init_mpc_height.repeat_interleave(4).reshape(-1, 4)

        # from src.envs.robots.modules.estimator.state_estimator import StateEstimator
        # self._state_estimator = StateEstimator(self)

        # subscribe to keyboard shortcuts
        self.subscribe_viewer_keyboard_event()

        self._post_physics_step()
        # self.reset()

    def _compute_base_init_state(self, init_positions: torch.Tensor):
        """Computes desired init state for CoM (position and velocity)."""
        num_envs = init_positions.shape[0]
        init_state_list = [0., 0., 0.] + [0., 0., 0., 1.] + [0., 0., 0.] + [0., 0., 0.]
        init_states = np.stack([init_state_list] * num_envs, axis=0)
        init_states = to_torch(init_states, device=self._device)
        init_states[:, :3] = init_positions
        return to_torch(init_states, device=self._device)

    def _cache_robot_rigid_body_indices(self):
        # Robot rigid body indices in IsaacGym
        self._feet_indices = torch.zeros(len(self._feet_names),
                                         dtype=torch.long,
                                         device=self._device,
                                         requires_grad=False)
        self._calf_indices = torch.zeros(len(self._calf_names),
                                         dtype=torch.long,
                                         device=self._device,
                                         requires_grad=False)
        self._thigh_indices = torch.zeros(len(self._thigh_names),
                                          dtype=torch.long,
                                          device=self._device,
                                          requires_grad=False)
        self._body_indices = torch.zeros(self._num_bodies - len(self._feet_names) -
                                         len(self._thigh_names) -
                                         len(self._calf_names),
                                         dtype=torch.long,
                                         device=self._device)

        for i in range(len(self._feet_names)):
            self._feet_indices[i] = self._gym.find_actor_rigid_body_index(
                self._envs[0], self._robot_actors[0], self._feet_names[i], gymapi.DOMAIN_ENV)

        for i in range(len(self._calf_names)):
            self._calf_indices[i] = self._gym.find_actor_rigid_body_index(
                self._envs[0], self._robot_actors[0], self._calf_names[i], gymapi.DOMAIN_ENV)

        for i in range(len(self._thigh_names)):
            self._thigh_indices[i] = self._gym.find_actor_rigid_body_index(
                self._envs[0], self._robot_actors[0], self._thigh_names[i], gymapi.DOMAIN_ENV)

        all_body_names = self._gym.get_actor_rigid_body_names(self._envs[0], self._robot_actors[0])
        self._body_names = []
        limb_names = self._thigh_names + self._calf_names + self._feet_names
        idx = 0
        # foot_name_ = ['FR_hip']
        for name in all_body_names:
            if name not in limb_names:
                self._body_indices[idx] = self._gym.find_actor_rigid_body_handle(
                    self._envs[0], self._robot_actors[0], name)
                idx += 1
                self._body_names.append(name)

        # print(f"all_body_names: {all_body_names}")
        # print(f"feet_indices: {self._feet_indices}")
        # print(f"calf_indices: {self._calf_indices}")
        # print(f"thigh_indices: {self._thigh_indices}")
        # print(f"body_indices: {self._body_indices}")
        # print(f"body_names: {self._body_names}")

    def _load_urdf(self, urdf_path):
        """Since IsaacGym does not allow separating the environment creation process from the actor creation process
        due to its low-level optimization mechanism (the engine requires prior knowledge of the number of actors in
        each env for performance optimization), we have integrated both into the Robot class. While this introduces
        some redundancy in the code, it adheres to the design principles of IsaacGym."""

        asset_root = os.path.dirname(urdf_path)
        asset_file = os.path.basename(urdf_path)
        asset_config = asset_options_config.get_config()
        self._robot_asset = self._gym.load_asset(self._sim, asset_root, asset_file,
                                                 asset_config.asset_options)
        self._num_dof = self._gym.get_asset_dof_count(self._robot_asset)
        self._num_bodies = self._gym.get_asset_rigid_body_count(self._robot_asset)

        spacing_x = 10.
        spacing_y = 10.
        spacing_z = 1.
        env_lower = gymapi.Vec3(-spacing_x, -spacing_y, 0.)
        env_upper = gymapi.Vec3(spacing_x, spacing_y, spacing_z)
        for i in range(self._num_envs):
            env_handle = self._gym.create_env(self._sim, env_lower, env_upper, int(np.sqrt(self._num_envs)))
            env_origin = self._gym.get_env_origin(env_handle)

            start_pose = gymapi.Transform()
            start_pose.p = gymapi.Vec3(*self._base_init_state[i, :3])
            # start_pose.r = gymapi.Quat(*self._base_init_state[i, 3:7])
            actor_handle = self._gym.create_actor(env_handle, self._robot_asset,
                                                  start_pose, f"robot", i,
                                                  asset_config.self_collisions, 0)
            # Add outdoor scene
            world_env = self._world_env_type(
                sim=self._sim,
                gym=self._gym,
                viewer=self._viewer,
                env_handle=env_handle,
                transform=env_origin
            )
            # Add camera sensor
            camera_sensor = RGBDCamera(
                robot=self,
                sim=self._sim,
                env=env_handle,
                viewer=self._viewer,
                attached_rigid_body_index_in_env=0,  # Attach camera to robot base
            )

            self._gym.enable_actor_dof_force_sensors(env_handle, actor_handle)
            self._envs.append(env_handle)
            self._world_envs.append(world_env)
            self._robot_actors.append(actor_handle)
            self._camera_sensors.append(camera_sensor)

        # For each environment simulate their unnecessary dynamics after initial urdf loading
        for world_env in self._world_envs:
            world_env.simulate_irrelevant_dynamics()

        # Cache robot rigid body indices
        self._cache_robot_rigid_body_indices()
        self._num_rigid_body_per_env = self._gym.get_env_rigid_body_count(self._envs[0])
        self._num_actor_per_env = self._gym.get_actor_count(self._envs[0])

    def set_foot_friction(self, friction_coef, env_id=0):
        rigid_shape_props = self._gym.get_actor_rigid_shape_properties(
            self._envs[env_id], self._robot_actors[env_id])
        for idx in range(len(rigid_shape_props)):
            rigid_shape_props[idx].friction = friction_coef
        self._gym.set_actor_rigid_shape_properties(self._envs[env_id],
                                                   self._robot_actors[env_id],
                                                   rigid_shape_props)
        # import pdb
        # pdb.set_trace()

    def set_foot_frictions(self, friction_coefs, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self._num_envs)
        friction_coefs = friction_coefs * np.ones(self._num_envs)
        for env_id, friction_coef in zip(env_ids, friction_coefs):
            self.set_foot_friction(friction_coef, env_id=env_id)

    def _init_buffers(self):
        # get gym GPU state tensors
        actor_root_state = self._gym.acquire_actor_root_state_tensor(self._sim)
        dof_state_tensor = self._gym.acquire_dof_state_tensor(self._sim)
        net_contact_forces = self._gym.acquire_net_contact_force_tensor(self._sim)
        rigid_body_state = self._gym.acquire_rigid_body_state_tensor(self._sim)
        dof_force = self._gym.acquire_dof_force_tensor(self._sim)
        jacobians = self._gym.acquire_jacobian_tensor(self._sim, "robot")
        # print(f"jacobian: {gymtorch.wrap_tensor(jacobians)}")

        # Obtain global robot indices
        for i in range(len(self._envs)):
            index = self._gym.get_actor_index(self._envs[i], self._robot_actors[i], gymapi.DOMAIN_SIM)
            self._robot_actors_global_indices.append(index)
            rigid_body_dict = self._gym.get_actor_rigid_body_dict(self._envs[i], self._robot_actors[i])
            for v in sorted(rigid_body_dict.values()):
                idx = self._gym.get_actor_rigid_body_index(self._envs[i], self._robot_actors[i], v, gymapi.DOMAIN_SIM)
                self._robot_rigid_body_global_indices.append(idx)

        # Wrap all tensors
        actor_root_state = gymtorch.wrap_tensor(actor_root_state)
        dof_state_tensor = gymtorch.wrap_tensor(dof_state_tensor)
        net_contact_forces = gymtorch.wrap_tensor(net_contact_forces)
        rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        dof_force = gymtorch.wrap_tensor(dof_force)
        jacobians = gymtorch.wrap_tensor(jacobians)

        self._gym.refresh_actor_root_state_tensor(self._sim)
        self._gym.refresh_net_contact_force_tensor(self._sim)
        self._gym.refresh_rigid_body_state_tensor(self._sim)
        self._gym.refresh_dof_force_tensor(self._sim)
        self._gym.refresh_dof_state_tensor(self._sim)
        self._gym.refresh_jacobian_tensors(self._sim)

        # Robot state buffers
        self._all_root_states = actor_root_state.clone()
        self._all_root_states[self._robot_actors_global_indices] = self._base_init_state
        self._root_states = actor_root_state[self._robot_actors_global_indices]

        self._dof_state = dof_state_tensor  # TODO: Here dof_state_tensor.clone() causes issue!!!
        self._rigid_body_state = rigid_body_state[:self._num_envs * self._num_rigid_body_per_env, :]
        self._motor_positions = self._dof_state.view(self._num_envs, self._num_dof, 2)[..., 0]
        self._motor_velocities = self._dof_state.view(self._num_envs, self._num_dof, 2)[..., 1]
        self._base_quat = self._root_states[:self._num_envs, 3:7]
        self._base_rot_mat = quat_to_rot_mat(self._base_quat)
        self._base_rot_mat_t = torch.transpose(self._base_rot_mat, 1, 2)

        self._contact_forces = (net_contact_forces.view(self._num_envs, -1, 3))  # shape: num_envs, num_bodies, xyz axis
        self._motor_torques = dof_force.view(self._num_envs, self._num_dof)
        self._jacobian = jacobians
        self._base_lin_vel_world = self._root_states[:self._num_envs, 7:10]
        self._base_ang_vel_world = self._root_states[:self._num_envs, 10:13]
        self._gravity_vec = torch.stack([to_torch([0., 0., 1.], device=self._device)] * self._num_envs)
        self._projected_gravity = torch.bmm(self._base_rot_mat_t, self._gravity_vec[:, :, None])[:, :, 0]
        self._foot_velocities = self._rigid_body_state.view(self._num_envs,
                                                            self._num_rigid_body_per_env, 13)[:, self._feet_indices,
                                7:10]
        self._foot_positions = self._rigid_body_state.view(self._num_envs,
                                                           self._num_rigid_body_per_env, 13)[:, self._feet_indices, 0:3]
        # Other useful buffers
        self._torques = torch.zeros(self._num_envs,
                                    self._num_dof,
                                    dtype=torch.float,
                                    device=self._device,
                                    requires_grad=False)

    def reset(self):
        self.reset_idx(torch.arange(self._num_envs, device=self._device))

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        env_ids_int32 = self._num_actor_per_env * env_ids.to(dtype=torch.int32)

        self._time_since_reset[env_ids] = 0
        self._last_timestamp[env_ids] = 0

        self._foot_positions_prev[env_ids, :] = self.foot_positions_in_base_frame[env_ids, :].clone()
        self._foot_positions_prev[env_ids, :, 2] = -(self._base_init_state[env_ids, 2].
                                                     repeat_interleave(4).reshape(-1, 4))
        # Reset root states:
        # all_root_state = self._all_root_states
        # self._root_states[env_ids] = self._base_init_state[env_ids]
        # print(f"self._root_states: {self._root_states}")
        # print(f"env_ids: {env_ids}")
        # self._gym.set_actor_root_state_tensor(
        #     self._sim, gymtorch.unwrap_tensor(self._all_root_states)
        # )
        self._gym.set_actor_root_state_tensor_indexed(
            self._sim,
            gymtorch.unwrap_tensor(self._all_root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )

        # Reset dofs
        self._motor_positions[env_ids] = to_torch(self._init_motor_angles,
                                                  device=self._device,
                                                  dtype=torch.float)
        self._motor_velocities[env_ids] = 0.

        self._gym.set_dof_state_tensor_indexed(
            self._sim, gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        if len(env_ids) == self._num_envs:
            self._gym.simulate(self._sim)

        self._post_physics_step()

    def step(self, action):
        for _ in range(self._sim_config.action_repeat):
            self._torques, _ = self.motor_group.convert_to_torque(
                action, self._motor_positions, self._motor_velocities)
            # time.sleep(1)
            self._gym.set_dof_actuation_force_tensor(
                self._sim, gymtorch.unwrap_tensor(self._torques))
            self._gym.simulate(self._sim)
            # if self._device == "cpu":
            self._gym.fetch_results(self._sim, True)
            self._gym.refresh_dof_state_tensor(self._sim)

            # self._update_foot_positions()  # Update foot positions
            self._time_since_reset += self._sim_config.sim_params.dt
            # self._state_estimator.update_ground_normal_vec()

            # print(f"self._state_estimator.ground_normal: {self._state_estimator.ground_normal}")
            # self._gravity_vec = to_torch(self._state_estimator.ground_normal, device=self._device).repeat(
            #     self._num_envs, 1)

        self._post_physics_step()

    def _post_physics_step(self):
        # Refresh all tensors
        self._gym.refresh_actor_root_state_tensor(self._sim)
        self._gym.refresh_net_contact_force_tensor(self._sim)
        self._gym.refresh_rigid_body_state_tensor(self._sim)
        self._gym.refresh_dof_state_tensor(self._sim)
        self._gym.refresh_dof_force_tensor(self._sim)
        self._gym.refresh_jacobian_tensors(self._sim)

        # Obtain and get the tensor to update
        actor_root_state = self._gym.acquire_actor_root_state_tensor(self._sim)
        rigid_body_state = self._gym.acquire_rigid_body_state_tensor(self._sim)
        actor_root_state = gymtorch.wrap_tensor(actor_root_state)
        rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)

        # Update robot actor root state and rigid_body_state
        self._root_states = actor_root_state[self._robot_actors_global_indices]
        self._rigid_body_state = rigid_body_state

        self._base_quat[:] = self._root_states[:self._num_envs, 3:7]
        self._base_rot_mat = quat_to_rot_mat(self._base_quat)
        self._base_rot_mat_t = torch.transpose(self._base_rot_mat, 1, 2)
        self._base_lin_vel_world = self._root_states[:self._num_envs, 7:10]
        self._base_ang_vel_world = self._root_states[:self._num_envs, 10:13]
        self._projected_gravity[:] = torch.bmm(self._base_rot_mat_t, self._gravity_vec[:, :, None])[:, :, 0]
        self._foot_velocities = self._rigid_body_state.view(self._num_envs,
                                                            self._num_rigid_body_per_env, 13)[:, self._feet_indices,
                                7:10]
        self._foot_positions = self._rigid_body_state.view(self._num_envs,
                                                           self._num_rigid_body_per_env, 13)[:, self._feet_indices, 0:3]
        # print("*******************************************************************************")
        # print(f"actor_root_state: {self._root_states}")
        # print(f"self._foot_positions: {self._foot_positions}")
        # print(f"self._foot_velocities: {self._foot_velocities}")
        # print("*******************************************************************************")

        # print(f"foot_positions: {self._foot_positions}")
        # _foot_pos = torch.zeros_like(self._foot_positions)
        # _foot_pos[:, 0] = torch.clone(self._foot_positions[:, 1])
        # _foot_pos[:, 1] = torch.clone(self._foot_positions[:, 0])
        # _foot_pos[:, 2] = torch.clone(self._foot_positions[:, 3])
        # _foot_pos[:, 3] = torch.clone(self._foot_positions[:, 2])
        # self._foot_positions = _foot_pos
        # print(f"foot_positions changed: {self._foot_positions}")
        # time.sleep(123)

    def _update_foot_positions(self):

        dt = self.time_since_reset - self._last_timestamp
        self._last_timestamp = self.time_since_reset
        # print(f"time_since_reset: {self.time_since_reset}")
        # print(f"last_timestamp: {self._last_timestamp}")
        # print(f"dt: {dt}")

        base_vel_body_frame = self.base_velocity_body_frame
        foot_contacts = self.foot_contacts.clone()
        foot_positions = self.foot_positions_in_base_frame.clone()
        for i in range(self._num_envs):
            for leg_id in range(4):
                if foot_contacts[i, leg_id]:
                    self._foot_positions_prev[i, leg_id] = foot_positions[i, leg_id]
                else:
                    self._foot_positions_prev[i, leg_id] -= base_vel_body_frame[i] * dt[i]

        print(f"foot_contact_history: {self._foot_positions_prev}")

    def set_robot_base_color(self, color, env_ids=torch.arange(0, 1, device='cuda:0')):
        base_rigid_body_idx = 0
        base_color = gymapi.Vec3(*color)
        for env_id in env_ids:
            idx = env_id.item()
            self._gym.set_rigid_body_color(self._envs[idx], self._robot_actors[idx], base_rigid_body_idx,
                                           gymapi.MESH_VISUAL, base_color)

    def get_motor_angles_from_foot_positions(self, foot_local_positions):
        raise NotImplementedError()

    def update_init_positions(self, env_ids, init_positions):
        self._base_init_state[env_ids] = self._compute_base_init_state(init_positions)

    @property
    def base_init_state(self):
        return self._base_init_state

    @property
    def base_position(self):
        base_position = torch.clone(self._root_states[:self._num_envs, :3])
        return base_position

    @property
    def base_position_world(self):
        return self._root_states[:self._num_envs, :3]

    @property
    def base_orientation_rpy(self):
        return angle_normalize(
            get_euler_zyx_from_quaternion(self._root_states[:self._num_envs, 3:7]))

    @property
    def base_orientation_quat(self):
        return self._root_states[:self._num_envs, 3:7]

    @property
    def projected_gravity(self):
        return self._projected_gravity

    @property
    def base_rot_mat(self):
        return self._base_rot_mat

    @property
    def base_rot_mat_t(self):
        return self._base_rot_mat_t

    @property
    def base_velocity_world_frame(self):
        return self._base_lin_vel_world

    @property
    def base_velocity_body_frame(self):
        return torch.bmm(self._base_rot_mat_t, self._root_states[:self._num_envs, 7:10, None])[:, :, 0]

    @property
    def base_angular_velocity_world_frame(self):
        return self._base_ang_vel_world

    @property
    def base_angular_velocity_body_frame(self):
        return torch.bmm(self._base_rot_mat_t, self._root_states[:self._num_envs, 10:13, None])[:, :, 0]

    @property
    def motor_positions(self):
        return torch.clone(self._motor_positions)

    @property
    def motor_velocities(self):
        return torch.clone(self._motor_velocities)

    @property
    def motor_torques(self):
        return torch.clone(self._torques)

    @property
    def foot_positions_in_base_frame(self):
        foot_positions_world_frame = self._foot_positions
        base_position_world_frame = self._root_states[:self._num_envs, :3]
        # num_env x 4 x 3
        foot_position = (foot_positions_world_frame - base_position_world_frame[:, None, :])
        return torch.matmul(self._base_rot_mat_t, foot_position.transpose(1, 2)).transpose(1, 2)

    @property
    def foot_positions_in_world_frame(self):
        return torch.clone(self._foot_positions)

    @property
    def foot_position_history(self):
        return self._foot_positions_prev

    @property
    def foot_height(self):
        return self._foot_positions[:, :, 2]

    @property
    def foot_velocities_in_base_frame(self):
        foot_vels = torch.bmm(self.all_foot_jacobian,
                              self.motor_velocities[:, :, None]).squeeze()
        return foot_vels.reshape((self._num_envs, 4, 3))

    @property
    def foot_velocities_in_world_frame(self):
        return self._foot_velocities

    @property
    def foot_contacts(self):
        return self._contact_forces[:, self._feet_indices, 2] > 1.

    @property
    def foot_contact_forces(self):
        return self._contact_forces[:, self._feet_indices, :]

    @property
    def calf_contacts(self):
        return self._contact_forces[:, self._calf_indices, 2] > 1.

    @property
    def calf_contact_forces(self):
        return self._contact_forces[:, self._calf_indices, :]

    @property
    def thigh_contacts(self):
        return self._contact_forces[:, self._thigh_indices, 2] > 1.

    @property
    def thigh_contact_forces(self):
        return self._contact_forces[:, self._thigh_indices, :]

    @property
    def has_body_contact(self):
        return torch.any(torch.norm(self._contact_forces[:, self._body_indices, :], dim=-1) > 1., dim=1)

    @property
    def has_dense_body_contact(self):
        dense_threshold = 100
        return torch.any(torch.norm(self._contact_forces[:, self._body_indices, :], dim=-1) > dense_threshold, dim=1)

    @property
    def hip_positions_in_body_frame(self):
        raise NotImplementedError()

    @property
    def all_foot_jacobian(self):
        rot_mat_t = self.base_rot_mat_t
        # print(f"rot_mat_t: {rot_mat_t}")
        # print(f"self._jacobian: {self._jacobian}")
        # print(f"self._jacobian: {self._jacobian.shape}")
        jacobian = torch.zeros((self._num_envs, 12, 12), device=self._device)
        jacobian[:, :3, :3] = torch.bmm(rot_mat_t, self._jacobian[:, 4, :3, 6:9])
        jacobian[:, 3:6, 3:6] = torch.bmm(rot_mat_t, self._jacobian[:, 8, :3, 9:12])
        jacobian[:, 6:9, 6:9] = torch.bmm(rot_mat_t, self._jacobian[:, 12, :3, 12:15])
        jacobian[:, 9:12, 9:12] = torch.bmm(rot_mat_t, self._jacobian[:, 16, :3, 15:18])
        return jacobian

    @property
    def mpc_body_height(self):
        return self.base_init_state[0, 2]

    @property
    def motor_group(self):
        return self._motors

    @property
    def env_handles(self):
        return self._envs

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def num_dof(self):
        return self._num_dof

    @property
    def device(self):
        return self._device

    @property
    def time_since_reset(self):
        return torch.clone(self._time_since_reset)

    @property
    def control_timestep(self):
        return self._sim_config.dt * self._sim_config.action_repeat

    @property
    def camera_sensor(self):
        return self._camera_sensors

    def subscribe_viewer_keyboard_event(self):
        # self._gym.subscribe_viewer_keyboard_event(self._viewer, gymapi.KEY_ESCAPE, "QUIT")
        self._gym.subscribe_viewer_keyboard_event(self._viewer, gymapi.KEY_V, "toggle_viewer_sync")
        self._gym.subscribe_viewer_keyboard_event(self._viewer, gymapi.KEY_F, "free_cam")
        for i in range(9):
            self._gym.subscribe_viewer_keyboard_event(self._viewer, getattr(gymapi, "KEY_" + str(i)), "lookat" + str(i))
        self._gym.subscribe_viewer_keyboard_event(self._viewer, gymapi.KEY_LEFT_BRACKET, "prev_id")
        self._gym.subscribe_viewer_keyboard_event(self._viewer, gymapi.KEY_RIGHT_BRACKET, "next_id")
        self._gym.subscribe_viewer_keyboard_event(self._viewer, gymapi.KEY_SPACE, "pause")
        self._gym.subscribe_viewer_keyboard_event(self._viewer, gymapi.KEY_W, "vx_plus")
        self._gym.subscribe_viewer_keyboard_event(self._viewer, gymapi.KEY_S, "vx_minus")
        self._gym.subscribe_viewer_keyboard_event(self._viewer, gymapi.KEY_A, "left_turn")
        self._gym.subscribe_viewer_keyboard_event(self._viewer, gymapi.KEY_D, "right_turn")
        self.free_cam = False
        self.lookat_id = 0
        self.lookat_vec = torch.tensor([-0, 2, 1], requires_grad=False, device=self.device)

    def render(self, sync_frame_time=True):
        if self._viewer:
            # check for window closed
            if self._gym.query_viewer_has_closed(self._viewer):
                sys.exit()

            # check for keyboard events
            for evt in self._gym.query_viewer_action_events(self._viewer):
                # if evt.action == "QUIT" and evt.value > 0:
                #     sys.exit()
                if evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

                if not self.free_cam:
                    for i in range(9):
                        if evt.action == "lookat" + str(i) and evt.value > 0:
                            # self.lookat(i)
                            self.lookat_id = i
                    if evt.action == "prev_id" and evt.value > 0:
                        self.lookat_id = (self.lookat_id - 1) % self.num_envs
                        # self.lookat(self.lookat_id)
                    if evt.action == "next_id" and evt.value > 0:
                        self.lookat_id = (self.lookat_id + 1) % self.num_envs
                        # self.lookat(self.lookat_id)
                    if evt.action == "vx_plus" and evt.value > 0:
                        self.commands[self.lookat_id, 0] += 0.2
                    if evt.action == "vx_minus" and evt.value > 0:
                        self.commands[self.lookat_id, 0] -= 0.2
                    if evt.action == "left_turn" and evt.value > 0:
                        self.commands[self.lookat_id, 3] += 0.5
                    if evt.action == "right_turn" and evt.value > 0:
                        self.commands[self.lookat_id, 3] -= 0.5
                if evt.action == "free_cam" and evt.value > 0:
                    self.free_cam = not self.free_cam
                    if self.free_cam:
                        self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)

                if evt.action == "pause" and evt.value > 0:
                    self.pause = True
                    while self.pause:
                        time.sleep(0.1)
                        self._gym.draw_viewer(self._viewer, self._sim, True)
                        for evt in self._gym.query_viewer_action_events(self._viewer):
                            if evt.action == "pause" and evt.value > 0:
                                self.pause = False
                        if self._gym.query_viewer_has_closed(self._viewer):
                            sys.exit()

            # mean_pos = torch.min(self.base_position_world,
            #                      dim=0)[0].cpu().numpy() + np.array([-2.5, 2.5, 2.5]) * 0.6
            # # mean_pos = torch.min(self.base_position_world,
            # #                      dim=0)[0].cpu().numpy() + np.array([0.5, -1., 0.])
            # target_pos = torch.mean(self.base_position_world,
            #                         dim=0).cpu().numpy() + np.array([0., 0., -0.5])
            # cam_pos = gymapi.Vec3(*mean_pos)
            # cam_target = gymapi.Vec3(*target_pos)
            # self._gym.viewer_camera_look_at(self._viewer, None, cam_pos, cam_target)

            # if self._device != "cpu":
            #     self._gym.fetch_results(self._sim, True)
            self._gym.fetch_results(self._sim, True)

            # step graphics
            self._gym.step_graphics(self._sim)
            self._gym.draw_viewer(self._viewer, self._sim, True)
            if sync_frame_time:
                self._gym.sync_frame_time(self._sim)

            self._gym.poll_viewer_events(self._viewer)

            # Record a video or not
            if self.record_video:
                _depth_img = self._camera_sensors[0].get_current_frame()
                self._frames.append(_depth_img)
                pass
