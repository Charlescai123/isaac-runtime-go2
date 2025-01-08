"""Abstract class for (vectorized) robots."""
import os
import sys
import time
from typing import Any, List

from isaacgym import gymtorch
from isaacgym.torch_utils import to_torch
import ml_collections
import numpy as np
import torch

from src.configs.defaults import asset_options as asset_options_config
from src.envs.robots.utils.rotation_utils import quat_to_rot_mat, get_euler_xyz_from_quaternion

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
        self._gym = gymapi.acquire_gym()
        self._sim = sim
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
        # init_positions[:, 0] = init_positions[:, 0] + self._scene_offset_x + 2  # Forward init pos
        init_positions[:, 0] = init_positions[:, 0] + self._scene_offset_x + 6  # (For testing)
        # init_positions[:, 0] = init_positions[:, 0] + self._scene_offset_x - 2  # Backward init pos
        # print(f"init_positions: {init_positions}")
        # time.sleep(123)

        self._base_init_state = self._compute_base_init_state(init_positions)
        self._init_motor_angles = self._motors.init_positions
        self._envs = []
        self._world_envs = []
        self._robot_actors = []
        self._robot_actors_global_indices = []
        self._robot_rigid_body_global_indices = []
        self._time_since_reset = torch.zeros(self._num_envs, device=self._device)

        self.record_video = True  # Record a video or not

        if "cuda" in self._device:
            torch._C._jit_set_profiling_mode(False)
            torch._C._jit_set_profiling_executor(False)

        self._load_urdf(urdf_path)
        self._gym.prepare_sim(self._sim)

        # # 初次渲染
        # gym = self._gym
        # gym.simulate(self._sim)
        # gym.fetch_results(sim, True)
        # gym.step_graphics(sim)
        # gym.draw_viewer(self._viewer, sim, True)
        #
        # # 持续渲染
        # while not gym.query_viewer_has_closed(self._viewer):
        #     gym.simulate(sim)
        #     gym.fetch_results(sim, True)
        #     gym.step_graphics(sim)
        #     gym.draw_viewer(self._viewer, sim, True)
        #
        # # 清理资源
        # gym.destroy_viewer(self._viewer)
        # gym.destroy_sim(sim)

        self._frames = []
        self._camera_handle = self.add_camera()
        self._init_buffers()

        # subscribe to keyboard shortcuts
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

        self._post_physics_step()
        # self.load_plane_asset()
        # time.sleep(123)
        # self.reset()

    def _compute_base_init_state(self, init_positions: torch.Tensor):
        """Computes desired init state for CoM (position and velocity)."""
        num_envs = init_positions.shape[0]
        init_state_list = [0., 0., 0.] + [0., 0., 0., 1.] + [0., 0., 0.] + [0., 0., 0.]
        # init_state_list = [0., 0., 0.] + [0., 0., 0.7071, 0.7071] + [0., 0., 0.
        #                                                      ] + [0., 0., 0.]
        # init_state_list = [0., 0., 0.] + [ 0.0499792, 0, 0, 0.9987503
        #                                       ] + [0., 0., 0.] + [0., 0., 0.]
        init_states = np.stack([init_state_list] * num_envs, axis=0)
        init_states = to_torch(init_states, device=self._device)
        init_states[:, :3] = init_positions
        # init_states[:, :3] = 0.5
        return to_torch(init_states, device=self._device)

    def _load_urdf(self, urdf_path):
        """Since Isaacgym does not allow separating the environment creation process from the actor creation process
        due to its low-level optimization mechanism (the engine requires prior knowledge of the number of actors in
        each env for performance optimization), we have integrated both into the Robot class. While this introduces
        some redundancy in the code, it adheres to the design principles of Isaacgym."""

        asset_root = os.path.dirname(urdf_path)
        asset_file = os.path.basename(urdf_path)
        asset_config = asset_options_config.get_config()
        self._robot_asset = self._gym.load_asset(self._sim, asset_root, asset_file,
                                                 asset_config.asset_options)
        self._num_dof = self._gym.get_asset_dof_count(self._robot_asset)
        self._num_bodies = self._gym.get_asset_rigid_body_count(self._robot_asset)

        spacing = 10.
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        for i in range(self._num_envs):
            env_handle = self._gym.create_env(self._sim, env_lower, env_upper,
                                              int(np.sqrt(self._num_envs)))
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
                env_handle=env_handle
            )

            self._gym.enable_actor_dof_force_sensors(env_handle, actor_handle)
            self._envs.append(env_handle)
            self._world_envs.append(world_env)
            self._robot_actors.append(actor_handle)

        # Robot indices in IsaacGym
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

        self._num_rigid_body_per_env = self._gym.get_env_rigid_body_count(self._envs[0])

        # print(f"all_body_names: {all_body_names}")
        # print(f"feet_indices: {self._feet_indices}")
        # print(f"calf_indices: {self._calf_indices}")
        # print(f"thigh_indices: {self._thigh_indices}")
        # print(f"body_indices: {self._body_indices}")
        # print(f"body_names: {self._body_names}")
        # time.sleep(123)

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

            idx1 = self._gym.get_actor_rigid_body_index(self._envs[i], self._robot_actors[i], self._feet_indices[0],
                                                        gymapi.DOMAIN_SIM)
            idx2 = self._gym.get_actor_rigid_body_index(self._envs[i], self._robot_actors[i], self._feet_indices[1],
                                                        gymapi.DOMAIN_SIM)
            idx3 = self._gym.get_actor_rigid_body_index(self._envs[i], self._robot_actors[i], self._feet_indices[2],
                                                        gymapi.DOMAIN_SIM)
            idx4 = self._gym.get_actor_rigid_body_index(self._envs[i], self._robot_actors[i], self._feet_indices[3],
                                                        gymapi.DOMAIN_SIM)
            print(f"idx1: {idx1}")
            print(f"idx2: {idx2}")
            print(f"idx3: {idx3}")
            print(f"idx4: {idx4}")

            rigid_body_dict = self._gym.get_actor_rigid_body_dict(self._envs[i], self._robot_actors[i])
            print(f"rigid_body_dict: {rigid_body_dict}")
            for v in sorted(rigid_body_dict.values()):
                idx = self._gym.get_actor_rigid_body_index(self._envs[i], self._robot_actors[i], v, gymapi.DOMAIN_SIM)
                self._robot_rigid_body_global_indices.append(idx)

        # Wrap all tensors
        actor_root_state = gymtorch.wrap_tensor(actor_root_state)
        # actor_root_state_robot = gymtorch.wrap_tensor(actor_root_state_robot)
        dof_state_tensor = gymtorch.wrap_tensor(dof_state_tensor)
        # net_contact_forces = gymtorch.wrap_tensor(net_contact_forces)
        # rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        net_contact_forces = gymtorch.wrap_tensor(net_contact_forces)
        rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        dof_force = gymtorch.wrap_tensor(dof_force)
        jacobians = gymtorch.wrap_tensor(jacobians)

        # res = self._gym.get_actor_rigid_body_states(self._envs[i], self._robot_actors[i], gymapi.STATE_ALL)
        # print(f"res: {res}")
        # rigid_body_state.append(res)
        # rigid_body_state = rigid_body_state.reshape(-1, 3)
        # print(f"rigid_body_state: {to_torch(rigid_body_state, device=self._device).shape}")
        print(f"self._robot_rigid_body_global_indices: {self._robot_rigid_body_global_indices}")
        print(f"actor_root_state: {actor_root_state}")
        print(f"actor_root_state.shape: {actor_root_state.shape}")
        print(f"actor_root_state.type: {type(actor_root_state)}")
        print(f"dof_state_tensor: {dof_state_tensor.shape}")
        print(f"net_contact_forces: {net_contact_forces}")
        print(f"net_contact_forces: {net_contact_forces.shape}")
        print(f"rigid_body_state: {rigid_body_state}")
        print(f"rigid_body_state: {rigid_body_state.shape}")
        print(f"rigid_body_state: {rigid_body_state.dtype}")
        print(f"dof_force: {dof_force.shape}")
        print(f"jacobians: {jacobians.shape}")
        # time.sleep(123)

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
        print(f"self._root_states: {self._root_states}")

        self._dof_state = dof_state_tensor
        self._rigid_body_state = rigid_body_state[:self._num_envs * self._num_rigid_body_per_env, :]
        self._motor_positions = self._dof_state.view(self._num_envs, self._num_dof, 2)[..., 0]
        self._motor_velocities = self._dof_state.view(self._num_envs, self._num_dof, 2)[..., 1]
        self._base_quat = self._root_states[:self._num_envs, 3:7]
        self._base_rot_mat = quat_to_rot_mat(self._base_quat)
        self._base_rot_mat_t = torch.transpose(self._base_rot_mat, 1, 2)
        # print(f"self._base_rot_mat_t: {self._base_rot_mat_t}")

        self._contact_forces = (net_contact_forces.view(self._num_envs, -1, 3))  # shape: num_envs, num_bodies, xyz axis
        self._motor_torques = dof_force.view(self._num_envs, self._num_dof)
        self._jacobian = jacobians
        self._base_lin_vel_world = self._root_states[:self._num_envs, 7:10]
        self._base_ang_vel_world = self._root_states[:self._num_envs, 10:13]
        self._gravity_vec = torch.stack([to_torch([0., 0., 1.], device=self._device)] * self._num_envs)
        self._projected_gravity = torch.bmm(self._base_rot_mat_t, self._gravity_vec[:, :, None])[:, :, 0]
        self._foot_velocities = self._rigid_body_state.view(self._num_envs,
                                                            self._num_rigid_body_per_env, 13)[:, self._feet_indices, 7:10]
        self._foot_positions = self._rigid_body_state.view(self._num_envs,
                                                           self._num_rigid_body_per_env, 13)[:, self._feet_indices, 0:3]
        print(f"self._base_quat: {self._base_quat}")
        print(f"self._base_rot_mat: {self._base_rot_mat}")
        print(f"self._base_lin_vel_world: {self._base_lin_vel_world}")
        print(f"self._base_ang_vel_world: {self._base_ang_vel_world}")

        print(f"self._num_bodies: {self._num_bodies}")
        # print(f"rigid_body_view: {self._rigid_body_state.view(self._num_envs, self._num_bodies, 13)}")
        print(f"self._rigid_body_state: {self._rigid_body_state}")
        print(f"self._feet_indices: {self._feet_indices}")
        print(f"self._foot_positions: {self._foot_positions}")
        print(f"self._foot_velocities: {self._foot_velocities}")
        print(f"self._rigid_body_state: {self._rigid_body_state.shape}")
        print(f"self._feet_indices: {self._feet_indices.shape}")
        print(f"self._foot_positions: {self._foot_positions.shape}")
        print(f"self._foot_velocities: {self._foot_velocities.shape}")
        # time.sleep(123)
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

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        # env_ids_int32 = torch.tensor([0, 1], dtype=torch.int32)
        self._time_since_reset[env_ids] = 0

        # Reset root states:
        # all_root_state = self._all_root_states
        # self._root_states[env_ids] = self._base_init_state[env_ids]
        # print(f"self._root_states: {self._root_states}")
        # print(f"env_ids: {env_ids}")
        self._gym.set_actor_root_state_tensor(
            self._sim, gymtorch.unwrap_tensor(self._all_root_states)
        )
        # self._gym.set_actor_root_state_tensor_indexed(
        #     self._sim, gymtorch.unwrap_tensor(self._root_states),
        #     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32)
        # )
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
        # time.sleep(123)

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
            self._time_since_reset += self._sim_config.sim_params.dt

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
        # self._rigid_body_state = rigid_body_state

        self._base_quat[:] = self._root_states[:self._num_envs, 3:7]
        self._base_rot_mat = quat_to_rot_mat(self._base_quat)
        self._base_rot_mat_t = torch.transpose(self._base_rot_mat, 1, 2)
        # print(f"self._base_rot_mat: {self._base_rot_mat}")
        # print(f"self._base_rot_mat_t: {self._base_rot_mat_t}")
        # time.sleep(123)
        self._base_lin_vel_world = self._root_states[:self._num_envs, 7:10]
        self._base_ang_vel_world = self._root_states[:self._num_envs, 10:13]
        self._projected_gravity[:] = torch.bmm(self._base_rot_mat_t,
                                               self._gravity_vec[:, :, None])[:, :, 0]
        self._foot_velocities = self._rigid_body_state.view(self._num_envs,
                                                            self._num_rigid_body_per_env, 13)[:, self._feet_indices, 7:10]
        self._foot_positions = self._rigid_body_state.view(self._num_envs,
                                                           self._num_rigid_body_per_env, 13)[:, self._feet_indices, 0:3]
        print("*******************************************************************************")
        print(f"actor_root_state: {self._root_states}")
        print(f"self._foot_positions: {self._foot_positions}")
        print(f"self._foot_velocities: {self._foot_velocities}")
        print("*******************************************************************************")

        # print(f"foot_positions: {self._foot_positions}")
        # _foot_pos = torch.zeros_like(self._foot_positions)
        # _foot_pos[:, 0] = torch.clone(self._foot_positions[:, 1])
        # _foot_pos[:, 1] = torch.clone(self._foot_positions[:, 0])
        # _foot_pos[:, 2] = torch.clone(self._foot_positions[:, 3])
        # _foot_pos[:, 3] = torch.clone(self._foot_positions[:, 2])
        # self._foot_positions = _foot_pos
        # print(f"foot_positions changed: {self._foot_positions}")
        # time.sleep(123)

    def get_motor_angles_from_foot_positions(self, foot_local_positions):
        raise NotImplementedError()

    def update_init_positions(self, env_ids, init_positions):
        self._base_init_state[env_ids] = self._compute_base_init_state(
            init_positions)

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
            get_euler_xyz_from_quaternion(self._root_states[:self._num_envs, 3:7]))

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
        # print(f"self._root_states: {self._root_states}")
        # print(f"self._base_rot_mat_t: {self._base_rot_mat_t}")
        # print(f"res: {torch.bmm(self._base_rot_mat_t, self._root_states[:, 7:10, None])[:, :, 0]}")
        return torch.bmm(self._base_rot_mat_t, self._root_states[:self._num_envs, 7:10,
                                               None])[:, :, 0]

    @property
    def base_angular_velocity_world_frame(self):
        return self._base_ang_vel_world

    @property
    def base_angular_velocity_body_frame(self):
        return torch.bmm(self._base_rot_mat_t, self._root_states[:self._num_envs, 10:13,
                                               None])[:, :, 0]

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
        foot_position = (foot_positions_world_frame -
                         base_position_world_frame[:, None, :])
        # return torch.matmul(self._base_rot_mat_t,
        #                     foot_position.transpose(1, 2)).transpose(1, 2)
        res = torch.matmul(self._base_rot_mat_t,
                           foot_position.transpose(1, 2)).transpose(1, 2)
        # print(f"res: {res}")
        return res

    @property
    def foot_positions_in_world_frame(self):
        return torch.clone(self._foot_positions)

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
        return torch.any(torch.norm(self._contact_forces[:, self._body_indices, :],
                                    dim=-1) > 1.,
                         dim=1)

    @property
    def hip_positions_in_body_frame(self):
        raise NotImplementedError()

    @property
    def all_foot_jacobian(self):
        rot_mat_t = self.base_rot_mat_t
        # print(f"rot_mat_t: {rot_mat_t}")
        # print(f"self._jacobian: {self._jacobian}")
        # print(f"self._jacobian: {self._jacobian.shape}")
        # time.sleep(123)
        jacobian = torch.zeros((self._num_envs, 12, 12), device=self._device)
        jacobian[:, :3, :3] = torch.bmm(rot_mat_t, self._jacobian[:, 4, :3, 6:9])
        jacobian[:, 3:6, 3:6] = torch.bmm(rot_mat_t, self._jacobian[:, 8, :3, 9:12])
        jacobian[:, 6:9, 6:9] = torch.bmm(rot_mat_t, self._jacobian[:, 12, :3, 12:15])
        jacobian[:, 9:12, 9:12] = torch.bmm(rot_mat_t, self._jacobian[:, 16, :3, 15:18])
        return jacobian

    @property
    def motor_group(self):
        return self._motors

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

    def get_pcd(self):

        # Array of RGB Colors, one per camera, for dots in the resulting
        # point cloud. Points will have a color which indicates which camera's
        # depth image created the point.
        color_map = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 1], [1, 0, 1]])

        # Render all of the image sensors only when we need their output here
        # rather than every frame.
        self._gym.render_all_camera_sensors(self._sim)
        color_map = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 1], [1, 0, 1]])
        points = []
        color = []
        print("Converting Depth images to point clouds. Have patience...")

        # print("Deprojecting from camera %d" % c)
        # Retrieve depth and segmentation buffer
        rgba_buffer = self._gym.get_camera_image(self._sim, self._envs[0], self._camera_handle, gymapi.IMAGE_COLOR)
        depth_buffer = self._gym.get_camera_image(self._sim, self._envs[0], self._camera_handle, gymapi.IMAGE_DEPTH)
        seg_buffer = self._gym.get_camera_image(self._sim, self._envs[0], self._camera_handle,
                                                gymapi.IMAGE_SEGMENTATION)

        # Get the camera view matrix and invert it to transform points from camera to world
        # space
        vinv = np.linalg.inv(np.matrix(self._gym.get_camera_view_matrix(self._sim, self._envs[0], self._camera_handle)))

        # Get the camera projection matrix and get the necessary scaling
        # coefficients for deprojection
        proj = self._gym.get_camera_proj_matrix(self._sim, self._envs[0], self._camera_handle)
        fu = 2 / proj[0, 0]
        fv = 2 / proj[1, 1]

        # Ignore any points which originate from ground plane or empty space
        # depth_buffer[seg_buffer == 0] = -10001
        cam_width = 1920
        cam_height = 1080
        centerU = cam_width / 2
        centerV = cam_height / 2
        for i in range(cam_width):
            for j in range(cam_height):
                # if depth_buffer[j, i] < -10000:
                #     continue
                # if seg_buffer[j, i] > 0:
                u = -(i - centerU) / (cam_width)  # image-space coordinate
                v = (j - centerV) / (cam_height)  # image-space coordinate
                d = depth_buffer[j, i]  # depth buffer value
                X2 = [d * fu * u, d * fv * v, d, 1]  # deprojection vector
                p2 = X2 * vinv  # Inverse camera view to get world coordinates
                points.append([p2[0, 2], p2[0, 0], p2[0, 1]])
                # color.append(c)

        # # use pptk to visualize the 3d point cloud created above
        # v = pptk.viewer(points, color)
        # v.color_map(color_map)
        # # Sets a similar view to the gym viewer in the PPTK viewer
        # v.set(lookat=[0, 0, 0], r=5, theta=0.4, phi=0.707)
        # print("Point Cloud Complete")
        # print(f"point: {points}")
        # print(f"point: {points.shape}")
        import open3d as o3d

        # Convert points and color to numpy arrays
        points = np.array(points)
        np.save("new_pcld", points)
        # print(f"points: {points}")
        # print(f"points: {points.shape}")
        # colors = np.array([color_map[c] for c in color])
        # colors = rgba_buffer[]
        rgba_image = np.frombuffer(rgba_buffer, dtype=np.uint8).reshape(1080, 1920, 4)
        rgb_image = rgba_image[:, :, :3]
        # colors = rgb_image.reshape(-1, *rgb_image[2:])

        # print(f"colors: {colors}")
        # print(f"colors: {colors.shape}")

        # Create Open3D PointCloud object
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        # point_cloud.colors = o3d.utility.Vector3dVector(colors)
        # o3d.io.write_point_cloud("output1.ply", point_cloud, write_ascii=True)

        print("Point cloud saved to 'output.ply'")
        # Visualize the point cloud
        # o3d.visualization.draw_geometries([point_cloud],
        #                                   zoom=0.8,
        #                                   front=[0.0, 0.0, -1.0],
        #                                   lookat=[0.0, 0.0, 0.0],
        #                                   up=[0.0, -1.0, 0.0])
        # print("Point Cloud Visualization Complete")
        # pcd_points = np.load("pcld_right.npy")
        # bev_img = birds_eye_point_cloud(pcd_points)
        # print("loaded!!!!")
        # time.sleep(123)
        # if show_plot:
        #     axarr[0, 0].imshow(rgb)
        #     axarr[0, 1].imshow(realDepthImg)
        #     axarr[1, 0].imshow(seg)
        #     axarr[1, 1].imshow(bev_img)
        #     plt.pause(0.1)

        # label_save_folder = '.'
        # bev_save_path = os.path.join(label_save_folder, f"bev_.png")
        # plt.imsave(bev_save_path, bev_img)

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
                            self.lookat(i)
                            self.lookat_id = i
                    if evt.action == "prev_id" and evt.value > 0:
                        self.lookat_id = (self.lookat_id - 1) % self.num_envs
                        self.lookat(self.lookat_id)
                    if evt.action == "next_id" and evt.value > 0:
                        self.lookat_id = (self.lookat_id + 1) % self.num_envs
                        self.lookat(self.lookat_id)
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

            # Get point cloud data
            s = time.time()
            # self.get_pcd()
            e = time.time()
            print(f"get pcd time: {e - s}")

            # Record a video or not
            if self.record_video:
                self._gym.render_all_camera_sensors(self._sim)
                self._gym.start_access_image_tensors(self._sim)

                color_image = self._gym.get_camera_image(self._sim, self._envs[0], self._camera_handle,
                                                         gymapi.IMAGE_COLOR)
                # depth_tensor = self._gym.get_camera_image(self._sim, self._envs[0], self._camera_handle,
                #                                          gymapi.IMAGE_DEPTH)
                depth_image_ = self._gym.get_camera_image_gpu_tensor(self._sim,
                                                                     self._envs[0],
                                                                     self._camera_handle,
                                                                     gymapi.IMAGE_DEPTH)

                torch_camera_depth_tensor = gymtorch.wrap_tensor(depth_image_)
                # Clamp depth values to the range [near_plane, far_plane]
                near_plane = 0.1
                far_plane = 10.0
                # torch_camera_depth_tensor = torch.clamp(torch_camera_depth_tensor, min=near_plane, max=far_plane)
                print(f"torch_camera_depth_tensor: {torch_camera_depth_tensor}")

                _depth_img = torch_camera_depth_tensor.clone().cpu().numpy()

                # depth_image = gymtorch.wrap_tensor(depth_image_)
                # depth_image = self.process_depth_image(depth_image, i)

                self._gym.end_access_image_tensors(self._sim)

                # for i in range(self.num_envs):
                #     depth_image_ = self.gym.get_camera_image_gpu_tensor(self.sim,
                #                                                         self.envs[i],
                #                                                         self.cam_handles[i],
                #                                                         gymapi.IMAGE_DEPTH)
                #
                #     depth_image = gymtorch.wrap_tensor(depth_image_)
                # depth_image = self.process_depth_image(depth_image, i)

                # init_flag = self.episode_length_buf <= 1
                # if init_flag[i]:
                #     self.depth_buffer[i] = torch.stack([depth_image] * self.cfg.depth.buffer_len, dim=0)
                # else:
                #     self.depth_buffer[i] = torch.cat(
                #         [self.depth_buffer[i, 1:], depth_image.to(self.device).unsqueeze(0)],
                #         dim=0)

                # self._gym.end_access_image_tensors(self._sim)

                # depth_image = np.array(depth_image.cpu(), copy=True).reshape((1920, 1080))
                # position, velocity = get_ball_state(env, sphere_handle)
                # print("pos:{} vel{} ".format(position, velocity))

                rgba_image = np.frombuffer(color_image, dtype=np.uint8).reshape(1080, 1920, 4)

                rgb_image = rgba_image[:, :, :3]
                # print(f"rgb_image: {rgb_image}")

                # time.sleep(1)
                # optical_flow_in_pixels = np.zeros(np.shape(optical_flow_image))
                # # Horizontal (u)
                # optical_flow_in_pixels[0, 0] = image_width * (optical_flow_image[0, 0] / 2 ** 15)
                # # Vertical (v)
                # optical_flow_in_pixels[0, 1] = image_height * (optical_flow_image[0, 1] / 2 ** 15)

                # self._frames.append(rgb_image)
                import cv2

                rgb_img = cv2.cvtColor(rgb_image, cv2.COLOR_RGBA2BGR)

                # cv2.imshow('RGB Image', rgb_img)

                # _depth_img[np.isinf(_depth_img)] = -256
                def replace_inf_with_second_smallest(depth_image):
                    """
                    Replace all `inf` values in a depth image with the second smallest finite value.

                    Args:
                        depth_image (np.ndarray): Input depth image (2D array).

                    Returns:
                        np.ndarray: Depth image with `inf` values replaced.
                    """
                    print(f"depth_image: {depth_image}")

                    # Flatten the array and filter finite values
                    finite_values = depth_image[np.isfinite(depth_image)]

                    if len(finite_values) < 2:
                        raise ValueError(
                            "The depth image does not have enough finite values to determine the second smallest.")

                    # Find the unique sorted finite values
                    unique_values = np.unique(finite_values)

                    if len(unique_values) < 2:
                        raise ValueError(
                            "The depth image does not have enough unique finite values for a valid replacement.")

                    # The second smallest value
                    second_smallest = unique_values[1]

                    # Replace `inf` values with the second smallest value
                    result = np.copy(depth_image)
                    result[np.isinf(result)] = second_smallest

                    return result

                _depth_img = replace_inf_with_second_smallest(_depth_img)

                depth_normalized = cv2.normalize(_depth_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                # depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                depth_colored = depth_normalized
                cv2.imshow('Depth Image', depth_colored)

                # combined_image = np.hstack((rgb_image, depth_colored))
                # cv2.imshow('Image', combined_image)
                cv2.waitKey(1)
                print(f"_depth_img: {_depth_img}")
                print(f"_depth_img: {_depth_img.shape}")

                print(f"depth_normalized: {depth_normalized}")
                print(f"depth_normalized: {depth_normalized.shape}")
                print(f"depth_colored: {depth_colored}")
                print(f"depth_colored: {depth_colored.shape}")
                is_all_zero = np.count_nonzero(depth_normalized) == 0
                if is_all_zero:
                    np.savetxt('depth_error.txt', _depth_img)
                    # np.savetxt('raw_depth_tensor.txt', depth_image_)
                    # torch.save(torch_camera_depth_tensor, 'raw_depth_tensor_error.pt')
                    # time.sleep(123)
                else:
                    # np.savetxt('depth.txt', _depth_img)
                    # torch.save(torch_camera_depth_tensor, 'raw_depth_tensor.pt')
                    pass
                self._frames.append(_depth_img)

    def get_vision_observation(self, return_label=False):
        width = 1080
        height = 1920
        # fov = 90
        # near_val = 0.1
        # far_val = 5

        proj_mat = self._gym.get_camera_proj_matrix(self._sim, self._envs[0], self._camera_handle)

        cam_transform = self._gym.get_camera_transform(self._sim, self._envs[0], self._camera_handle)
        cam_pos = cam_transform.p
        cam_orn = cam_transform.r

        view_mat2 = self._gym.get_camera_view_matrix(self._sim, self._envs[0], self._camera_handle)

        self._gym.render_all_camera_sensors(self._sim)

        self._gym.render_all_camera_sensors(self._sim)
        self._gym.start_access_image_tensors(self._sim)

        color_image = self._gym.get_camera_image(self._sim, self._envs[0], self._camera_handle,
                                                 gymapi.IMAGE_COLOR)
        # depth_tensor = self._gym.get_camera_image(self._sim, self._envs[0], self._camera_handle,
        #                                          gymapi.IMAGE_DEPTH)
        depth_image_ = self._gym.get_camera_image_gpu_tensor(self._sim,
                                                             self._envs[0],
                                                             self._camera_handle,
                                                             gymapi.IMAGE_DEPTH)

        torch_camera_depth_tensor = gymtorch.wrap_tensor(depth_image_)
        # Clamp depth values to the range [near_plane, far_plane]
        near_plane = 0.1
        far_plane = 10.0
        # torch_camera_depth_tensor = torch.clamp(torch_camera_depth_tensor, min=near_plane, max=far_plane)
        print(f"torch_camera_depth_tensor: {torch_camera_depth_tensor}")

        _depth_img = torch_camera_depth_tensor.clone().cpu().numpy()

        # depth_image = gymtorch.wrap_tensor(depth_image_)
        # depth_image = self.process_depth_image(depth_image, i)

        self._gym.end_access_image_tensors(self._sim)

        # for i in range(self.num_envs):
        #     depth_image_ = self.gym.get_camera_image_gpu_tensor(self.sim,
        #                                                         self.envs[i],
        #                                                         self.cam_handles[i],
        #                                                         gymapi.IMAGE_DEPTH)
        #
        #     depth_image = gymtorch.wrap_tensor(depth_image_)
        # depth_image = self.process_depth_image(depth_image, i)

        # init_flag = self.episode_length_buf <= 1
        # if init_flag[i]:
        #     self.depth_buffer[i] = torch.stack([depth_image] * self.cfg.depth.buffer_len, dim=0)
        # else:
        #     self.depth_buffer[i] = torch.cat(
        #         [self.depth_buffer[i, 1:], depth_image.to(self.device).unsqueeze(0)],
        #         dim=0)

        # self._gym.end_access_image_tensors(self._sim)

        # depth_image = np.array(depth_image.cpu(), copy=True).reshape((1920, 1080))
        # position, velocity = get_ball_state(env, sphere_handle)
        # print("pos:{} vel{} ".format(position, velocity))

        rgba_image = np.frombuffer(color_image, dtype=np.uint8).reshape(1080, 1920, 4)

        rgb_image = rgba_image[:, :, :3]
        # print(f"rgb_image: {rgb_image}")

        # time.sleep(1)
        # optical_flow_in_pixels = np.zeros(np.shape(optical_flow_image))
        # # Horizontal (u)
        # optical_flow_in_pixels[0, 0] = image_width * (optical_flow_image[0, 0] / 2 ** 15)
        # # Vertical (v)
        # optical_flow_in_pixels[0, 1] = image_height * (optical_flow_image[0, 1] / 2 ** 15)

        # self._frames.append(rgb_image)
        import cv2

        # rgb_img = cv2.cvtColor(rgb_image, cv2.COLOR_RGBA2BGR)
        # cv2.imshow('RGB Image', rgb_img)

        # _depth_img[np.isinf(_depth_img)] = -256

        # cv2.imshow('Depth Image', depth_colored)
        # cv2.waitKey(1)
        # print(f"_depth_img: {_depth_img}")
        # print(f"_depth_img: {_depth_img.shape}")
        #
        # print(f"depth_normalized: {depth_normalized}")
        # print(f"depth_normalized: {depth_normalized.shape}")
        # print(f"depth_colored: {depth_colored}")
        # print(f"depth_colored: {depth_colored.shape}")
        # is_all_zero = np.count_nonzero(depth_normalized) == 0

        # color_img = self._gym.get_camera_image(self._sim, self._envs[0], self._camera_handle, gymapi.IMAGE_COLOR)
        # depth_img = self._gym.get_camera_image(self._sim, self._envs[0], self._camera_handle, gymapi.IMAGE_DEPTH)

        seg_img = self._gym.get_camera_image(self._sim, self._envs[0], self._camera_handle, gymapi.IMAGE_SEGMENTATION)

        self._gym.render_all_camera_sensors(self._sim)
        self._gym.start_access_image_tensors(self._sim)

        color_image = self._gym.get_camera_image(self._sim, self._envs[0], self._camera_handle, gymapi.IMAGE_COLOR)
        # depth_tensor = self._gym.get_camera_image(self._sim, self._envs[0], self._camera_handle,
        #                                          gymapi.IMAGE_DEPTH)
        depth_image_ = self._gym.get_camera_image_gpu_tensor(self._sim,
                                                             self._envs[0],
                                                             self._camera_handle,
                                                             gymapi.IMAGE_DEPTH)

        torch_camera_depth_tensor = gymtorch.wrap_tensor(depth_image_)

        _depth_img = torch_camera_depth_tensor.clone().cpu().numpy()

        # depth_image = gymtorch.wrap_tensor(depth_image_)
        # depth_image = self.process_depth_image(depth_image, i)

        self._gym.end_access_image_tensors(self._sim)

        color_img = color_image
        rgba_img = np.frombuffer(color_img, dtype=np.uint8).reshape(1080, 1920, 4)

        # rgb_img = rgba_img[:, :, :3]
        # print(f"rgb_image: {rgb_image}")

        # time.sleep(1)
        # optical_flow_in_pixels = np.zeros(np.shape(optical_flow_image))
        # # Horizontal (u)
        # optical_flow_in_pixels[0, 0] = image_width * (optical_flow_image[0, 0] / 2 ** 15)
        # # Vertical (v)
        # optical_flow_in_pixels[0, 1] = image_height * (optical_flow_image[0, 1] / 2 ** 15)

        if return_label:
            info = {"cam_pos": cam_pos, "cam_orn": cam_orn,
                    "view_matrix": view_mat2,
                    "projection_matrix": proj_mat,
                    "width": width,
                    "height": height}
        else:
            info = None

        depth_normalized = cv2.normalize(_depth_img, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
        # depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        np.savetxt("depth_normalized.txt", depth_normalized, fmt="%.6f")
        print(f"rgb_img: {rgba_img}")
        print(f"rgb_img: {rgba_img.shape}")
        print(f"depth: {_depth_img}")
        print(f"depth: {_depth_img.shape}")
        print(f"depth normalized: {depth_normalized}")
        print(f"depth normalized: {depth_normalized.shape}")
        # time.sleep(123)
        return rgba_img, depth_normalized, seg_img, info

    def add_camera(self):
        # create camera
        camera_props = gymapi.CameraProperties()
        camera_props.width = 1920
        camera_props.height = 1080
        camera_props.enable_tensors = True  # Enable tensor output for the camera
        camera_props.near_plane = 0.1  # Minimum distance
        camera_props.far_plane = 10.0  # Maximum distance
        camera_horizontal_fov = 87
        camera_props.horizontal_fov = camera_horizontal_fov
        camera_handle = self._gym.create_camera_sensor(self._envs[0], camera_props)
        camera_pos = [0, 0, 2]

        local_transform = gymapi.Transform()
        local_transform.p = gymapi.Vec3(*camera_pos)
        local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.radians(30.0))
        self._gym.attach_camera_to_body(camera_handle, self._envs[0], self._robot_actors[0], local_transform,
                                        gymapi.FOLLOW_TRANSFORM)

        return camera_handle

    def create_recording_camera(self, gym, env_handle,
                                resolution=(1920, 1080),
                                h_fov=86,
                                actor_to_attach=None,
                                transform=None,  # related to actor_to_attach
                                ):
        camera_props = gymapi.CameraProperties()
        camera_props.enable_tensors = True
        camera_props.width = resolution[0]
        camera_props.height = resolution[1]
        camera_props.horizontal_fov = h_fov
        camera_handle = gym.create_camera_sensor(env_handle, camera_props)
        if actor_to_attach is not None:
            gym.attach_camera_to_body(
                camera_handle,
                env_handle,
                actor_to_attach,
                transform,
                gymapi.FOLLOW_POSITION,
            )
        elif transform is not None:
            gym.set_camera_transform(
                camera_handle,
                env_handle,
                transform,
            )
        return camera_handle
