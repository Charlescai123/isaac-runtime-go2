import os
import sys
import time

from isaacgym import gymapi
from src.configs.defaults import asset_options as asset_options_config
from src.envs.terrains.utils import load_cement_road_asset, load_stone_asset
from src.envs.terrains.utils import load_random_cobblestones_in_a_region
from src.envs.terrains.utils import load_random_snowstones_in_a_region, add_uneven_terrains

from typing import Any, List
import ml_collections
import numpy as np
import torch


class WildTerrainEnv:
    def __init__(
            self,
            sim: Any,
            gym: Any,
            viewer: Any,
            num_envs: int,
            env_handles: List,
            sim_config: ml_collections.ConfigDict
    ):
        """Initialize the wild terrain env class."""
        self._sim = sim
        self._gym = gym
        self._viewer = viewer
        self._enable_viewer_sync = True
        self._sim_config = sim_config
        self._device = self._sim_config.sim_device
        self._envs = env_handles
        self._num_envs = num_envs

        self._envs = []
        self._actors = []

        if "cuda" in self._device:
            torch._C._jit_set_profiling_mode(False)
            torch._C._jit_set_profiling_executor(False)

        self._load_all_assets()
        # self._gym.prepare_sim(self._sim)

        self._post_physics_step()

    def _init_buffers(self):
        pass

    def _load_all_assets(self):
        for env_handle in self._envs:
            # Add outdoor scene
            self.load_outdoor_asset(env=env_handle, reverse=False)

        # Add uneven terrains
        add_uneven_terrains(gym=self._gym, sim=self._sim, reverse=False)

    def load_outdoor_asset(self, env, scene_offset_x=40, reverse=False):
        # initial root pose for actors
        initial_pose = gymapi.Transform()
        initial_pose.p = gymapi.Vec3(6.0, -2.0, 0.)

        if reverse:
            offset_x = -28 + scene_offset_x
        else:
            offset_x = 0 + scene_offset_x
        offset_y = 4

        # Directional Arrow
        # load_arrow_asset(gym=gym, sim=sim, i=0, pos=(3 + offset_x, -2 + offset_y, 0.), rot=(1, 0, 0, 0),
        #                  apply_texture=False, scale=1.2)
        # load_arrow_asset(gym=gym, sim=sim, i=1, pos=(3 + offset_x, -6 + offset_y, 0.), rot=(1, 0, 0, 0),
        #                  apply_texture=True, scale=1.2)

        # Cement road
        if reverse:
            load_cement_road_asset(self._gym, self._sim, env=env, pos=(-5 + scene_offset_x, -4 + offset_y, 0.001),
                                   apply_texture=True, scale=1.0)
        else:
            load_cement_road_asset(self._gym, self._sim, env=env, pos=(4 + scene_offset_x + 1, -4 + offset_y, 0.001),
                                   apply_texture=True, scale=1.0)

        # Big Snow Rocks
        load_stone_asset(self._gym, self._sim, env=env, pos=(12 + offset_x, -0.5 + offset_y, 0.1), rot=(0, 0, 1, 0),
                         apply_texture=False, scale=0.25)
        load_stone_asset(self._gym, self._sim, env=env, pos=(14 + offset_x, offset_y, 0.1), rot=(0.3, 0.2, 0.1, 1),
                         apply_texture=False, scale=0.3)
        load_stone_asset(self._gym, self._sim, env=env, pos=(17 + offset_x, -1 + offset_y, 0.1), rot=(0.2, 0.1, 0.8, 0),
                         apply_texture=False, scale=0.35)
        load_stone_asset(self._gym, self._sim, env=env, pos=(10 + offset_x, offset_y, 0.1), rot=(0., 0., 1., 0.),
                         apply_texture=False, scale=0.4)
        load_stone_asset(self._gym, self._sim, env=env, pos=(18 + offset_x, -8 + offset_y, 0.1), rot=(0, 0, 0, 1),
                         apply_texture=False, scale=0.4)
        load_stone_asset(self._gym, self._sim, env=env, pos=(10 + offset_x, -7 + offset_y, 0.3), rot=(0.2, 0, 0.8, 0),
                         apply_texture=False, scale=0.3)
        load_stone_asset(self._gym, self._sim, env=env, pos=(13 + offset_x, -8.5 + offset_y, 0.1), rot=(0.2, 0, 0.8, 0),
                         apply_texture=False, scale=0.55)
        load_stone_asset(self._gym, self._sim, env=env, pos=(10 + offset_x, -3 + offset_y, 0.1), rot=(0.2, 0, 0.8, 0),
                         apply_texture=False, scale=0.25)

        # Set up random stones
        load_random_snowstones_in_a_region(self._gym, self._sim, env=env, stone_nums=450, reverse=reverse)
        load_random_cobblestones_in_a_region(self._gym, self._sim, env=env, stone_nums=150, reverse=reverse)

        # Christmas Tree
        # initial_pose1 = gymapi.Transform()
        # initial_pose2 = gymapi.Transform()
        # initial_pose1.p = gymapi.Vec3(6.0 + offset_x, -2.0 + offset_y, 0.)
        # initial_pose2.p = gymapi.Vec3(6.0 + offset_x, -6.0 + offset_y, 0.)
        # asset3 = gym.load_asset(sim, f"{asset_root}/data", "tree/tree.urdf", asset_options)
        # env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
        # actor3 = gym.create_actor(env, asset3, initial_pose1, 'actor_tree1', 0, 1)
        # env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
        # actor4 = gym.create_actor(env, asset3, initial_pose2, 'actor_tree2', 1, 1)

        texture_handle = self._gym.create_texture_from_file(self._sim, "meshes/quad_gym/env/assets/grass.png")

        # Apply texture to Box Actor
        # gym.set_rigid_body_texture(env, actor2, 0, gymapi.MESH_VISUAL, texture_handle)
