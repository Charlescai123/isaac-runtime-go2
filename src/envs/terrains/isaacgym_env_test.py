import os
import sys
import time
import numpy as np
from isaacgym import gymapi, gymutil
import torch

from src.envs.terrains.wild_env_test1 import WildTerrainEnvTest1


class IsaacGymTester:
    def __init__(self, use_viewer=True, use_camera=True, device="cuda"):
        self.use_viewer = use_viewer
        self.use_camera = use_camera
        self.device = device

        # 初始化 Isaac Gym
        self.gym = gymapi.acquire_gym()

        # 设置 Sim 参数
        self.sim = self.create_sim()
        self.envs = []
        self.actors = []
        self.cameras = []
        self._world_envs = []

        # 创建 Viewer（可选）
        self.viewer = None
        if self.use_viewer:
            self.viewer = self.create_viewer()

        self._create_terrain()
        self._load_urdf()

    def _create_terrain(self):
        """Creates terrains.

        Note that we set the friction coefficient to all 0 here. This is because
        Isaac seems to pick the larger friction out of a contact pair as the
        actual friction coefficient. We will set the corresponding friction coefficient
        in robot friction.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = .2
        plane_params.dynamic_friction = .2
        plane_params.restitution = 0.
        self.gym.add_ground(self.sim, plane_params)
        self._terrain = None

    def _load_urdf(self):
        """Since IsaacGym does not allow separating the environment creation process from the actor creation process
        due to its low-level optimization mechanism (the engine requires prior knowledge of the number of actors in
        each env for performance optimization), we have integrated both into the Robot class. While this introduces
        some redundancy in the code, it adheres to the design principles of IsaacGym."""

        # asset_root = os.path.dirname(urdf_path)
        # asset_file = os.path.basename(urdf_path)
        # asset_config = asset_options_config.get_config()
        # self._robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file,
        #                                         asset_config.asset_options)
        # self._num_dof = self.gym.get_asset_dof_count(self._robot_asset)
        # self._num_bodies = self.gym.get_asset_rigid_body_count(self._robot_asset)

        spacing_x = 10.
        spacing_y = 10.
        spacing_z = 1.
        env_lower = gymapi.Vec3(-spacing_x, -spacing_y, 0.)
        env_upper = gymapi.Vec3(spacing_x, spacing_y, spacing_z)
        env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(1)))
        env_origin = self.gym.get_env_origin(env_handle)

        # Add outdoor scene
        world_env = WildTerrainEnvTest1(
            sim=self.sim,
            gym=self.gym,
            viewer=self.viewer,
            env_handle=env_handle,
            transform=env_origin
        )

        self.envs.append(env_handle)
        self._world_envs.append(world_env)

        # For each environment simulate their unnecessary dynamics after initial urdf loading
        for world_env in self._world_envs:
            world_env.simulate_irrelevant_dynamics()

    def create_sim(self):
        """ 创建仿真环境 """
        sim_params = gymapi.SimParams()
        sim_params.substeps = 2
        sim_params.dt = 1 / 60.0  # 60Hz
        sim_params.up_axis = gymapi.UP_AXIS_Z  # 以 Z 轴为上方向
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)  # 重力

        # 物理引擎选择
        sim_params.physx.use_gpu = self.device == "cuda"

        # 创建仿真器
        sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        if sim is None:
            raise RuntimeError("Failed to create sim")
        return sim

    def create_viewer(self):
        """ 创建可视化窗口 """
        viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if viewer is None:
            raise RuntimeError("Failed to create viewer")
        return viewer

    def create_env(self, env_index, env_size=5.0):
        """ 创建一个环境 """
        lower = gymapi.Vec3(-env_size, -env_size, 0.0)
        upper = gymapi.Vec3(env_size, env_size, env_size)
        env = self.gym.create_env(self.sim, lower, upper, 1)
        self.envs.append(env)
        return env

    def load_actor(self, env, urdf_path, position=(0, 0, 0.5)):
        """ 加载 URDF/机器人 actor """
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False  # 可移动
        asset_options.use_mesh_materials = True

        # 加载 URDF 资产
        asset = self.gym.load_asset(self.sim, os.path.dirname(urdf_path), os.path.basename(urdf_path), asset_options)

        # 设置初始位置
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*position)
        pose.r = gymapi.Quat(0, 0, 0, 1)

        # 创建 actor
        actor_handle = self.gym.create_actor(env, asset, pose, "robot", env_index, 1)
        self.actors.append(actor_handle)
        return actor_handle

    def add_camera(self, env, position=(1, 1, 1), look_at=(0, 0, 0)):
        """ 添加相机 """
        cam_props = gymapi.CameraProperties()
        cam_handle = self.gym.create_camera_sensor(env, cam_props)
        self.gym.set_camera_location(cam_handle, env, gymapi.Vec3(*position), gymapi.Vec3(*look_at))
        self.cameras.append(cam_handle)
        return cam_handle

    def step_simulation(self):
        """ 主仿真循环 """
        while self.use_viewer and not self.gym.query_viewer_has_closed(self.viewer):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)

            # 渲染相机
            if self.use_camera:
                self.gym.render_all_camera_sensors(self.sim)

            self.gym.sync_frame_time(self.sim)

    def cleanup(self):
        """ 清理资源 """
        if self.viewer:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def render(self):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            self.gym.fetch_results(self.sim, True)

            # step graphics
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)

            self.gym.sync_frame_time(self.sim)

            self.gym.poll_viewer_events(self.viewer)


# ========== 运行测试 ========== #
if __name__ == "__main__":
    tester = IsaacGymTester(use_viewer=True, use_camera=True)

    # 创建环境
    env = tester.create_env(env_index=0)

    # 加载机器人（可修改 URDF 路径）
    urdf_path = "/path/to/your/robot.urdf"  # 修改为你的 URDF 路径
    # actor = tester.load_actor(env, urdf_path)

    # 添加相机
    cam_handle = tester.add_camera(env, position=(2, 2, 2), look_at=(0, 0, 0))

    step = 0
    # 开始仿真
    try:
        tester.step_simulation()
        tester.render()
        print(f"step: {step}")
        step += 1
    except KeyboardInterrupt:
        print("Simulation stopped.")
    finally:
        tester.cleanup()
