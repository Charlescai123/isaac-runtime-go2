import time

from isaacgym import gymapi, gymtorch
from src.envs.terrains.utils import load_cement_road_asset, load_stone_asset, load_stop_sign_asset, \
    load_snow_tree_asset, load_oak_tree_asset, load_dead_wood_asset
from src.envs.terrains.utils import random_quaternion, add_uneven_terrains

import numpy as np
from typing import Any


class WildTerrainEnvTest1:
    def __init__(
            self,
            sim: Any,
            gym: Any,
            viewer: Any,
            env_handle: Any,
            transform: gymapi.Transform()
    ):
        """Initialize the wild terrain env class."""

        self._sim = sim
        self._gym = gym
        self._viewer = viewer
        self._enable_viewer_sync = True
        # self._sim_config = sim_config
        # self._device = self._sim_config.sim_device
        self._actors = []
        self._env = env_handle
        self._transform = transform

        # if "cuda" in self._device:
        #     torch._C._jit_set_profiling_mode(False)
        #     torch._C._jit_set_profiling_executor(False)

        self._load_all_assets()

        # Simulate unnecessary terrain dynamics
        # self.simulate_irrelevant_dynamics()

    def simulate_irrelevant_dynamics(self, step_cnt=1000):
        """Simulate and bypass irrelevant terrain dynamics initially to prevent unwanted motion in subsequent robot
        interactions."""
        self._gym.prepare_sim(self._sim)
        for _ in range(step_cnt):
            self._gym.simulate(self._sim)
            # self._gym.fetch_results(self._sim, True)
            # self._gym.step_graphics(self._sim)
            # self._gym.draw_viewer(self._viewer, self._sim, True)

    def _load_all_assets(self):
        print(f"loading all assets....")
        offset_x = 40
        # offset_x = 0
        # Add outdoor scene
        self.load_outdoor_asset(env=self._env, scene_offset_x=offset_x, reverse=False)
        # Add uneven terrains
        add_uneven_terrains(gym=self._gym, sim=self._sim, scene_offset_x=offset_x + self._transform.x,
                            scene_offset_y=self._transform.y, reverse=False)

    def load_outdoor_asset(self, env, scene_offset_x=0, reverse=False):
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

        # Cement Road
        if reverse:
            actor = load_cement_road_asset(self._gym, self._sim, env=env, name="cement_road", fix_base_link=True,
                                           pos=(-5 + scene_offset_x, -4 + offset_y, 0.001), apply_texture=True,
                                           scale=1.0)
        else:
            actor = load_cement_road_asset(self._gym, self._sim, env=env, name="cement_road", fix_base_link=True,
                                           pos=(4 + scene_offset_x + 1, -4 + offset_y, 0.001), apply_texture=True,
                                           scale=1.0)
        self._actors.append(actor)

        # Mountain Rocks
        self.load_mountain_rocks(env=env, offset_x=offset_x, offset_y=offset_y)

        # Snow Rocks
        self.load_snow_rocks(env=env, offset_x=offset_x, offset_y=offset_y)

        # Random stones (movable stones)
        self.load_random_snowstones_in_a_region(env=env, scene_offset_x=offset_x, width=10, length=10, stone_nums=600,
                                                reverse=reverse)
        # Random stones (static stones)
        self.load_random_cobblestones_in_a_region(env=env, scene_offset_x=offset_x, stone_nums=100, reverse=reverse)

        # Stopsign
        # self.load_stop_sign(env=env, offset_x=offset_x, offset_y=offset_y)

        # Snow Tree
        self.load_snow_tree(env=env, offset_x=offset_x, offset_y=offset_y)

        # Oak Tree
        self.load_oak_tree(env=env, offset_x=offset_x, offset_y=offset_y)

        # Dead wood
        self.load_dead_wood(env=env, offset_x=offset_x, offset_y=offset_y)

    def load_mountain_rocks(self, env, offset_x, offset_y):
        offset_x -= 7
        offset_y = 0
        # Big Mountain Rocks
        actor1 = load_stone_asset(self._gym, self._sim, env=env, name="Mountain Rock1", fix_base_link=False,
                                  pos=(25 + offset_x, -1.5 + offset_y, 0.1), rot=(0, 0, 1, 0), scale=1.25)
        actor2 = load_stone_asset(self._gym, self._sim, env=env, name="Mountain Rock2", fix_base_link=False,
                                  pos=(14.5 + offset_x, 2 + offset_y, 0.1), rot=(0.3, 0.2, 0.1, 1), scale=1.3)
        actor3 = load_stone_asset(self._gym, self._sim, env=env, name="Mountain Rock3", fix_base_link=False,
                                  pos=(19 + offset_x, 2.8 + offset_y, 0.1), rot=(0.2, 0.1, 0.8, 0), scale=1.55)
        actor4 = load_stone_asset(self._gym, self._sim, env=env, name="Mountain Rock4", fix_base_link=False,
                                  pos=(16 + offset_x, -.5 + offset_y, 0.1), rot=(0., 0., 1., 0.), scale=1.4)
        actor5 = load_stone_asset(self._gym, self._sim, env=env, name="Mountain Rock5", fix_base_link=False,
                                  pos=(15 + offset_x, -2 + offset_y, 0.1), rot=(0, 0, 0, 1), scale=1.45)
        actor6 = load_stone_asset(self._gym, self._sim, env=env, name="Mountain Rock6", fix_base_link=False,
                                  pos=(21 + offset_x, -.2 + offset_y, 0.1), rot=(0.1, 0.2, 0.2, 0), scale=1.7)
        actor7 = load_stone_asset(self._gym, self._sim, env=env, name="Mountain Rock7", fix_base_link=False,
                                  pos=(17.5 + offset_x, -2.5 + offset_y, 0.1), rot=(0.2, 0, 0.8, 0), scale=1.55)
        actor8 = load_stone_asset(self._gym, self._sim, env=env, name="Mountain Rock8", fix_base_link=False,
                                  pos=(17 + offset_x, 3.2 + offset_y, 0.1), rot=(0.2, 0, 0.8, 0), scale=1.25)
        actor9 = load_stone_asset(self._gym, self._sim, env=env, name="Mountain Rock9", fix_base_link=False,
                                  pos=(20.5 + offset_x, 1 + offset_y, 0.1), rot=(0.3, -0.2, -0.8, 0), scale=1.65)
        actor10 = load_stone_asset(self._gym, self._sim, env=env, name="Mountain Rock10", fix_base_link=False,
                                   pos=(21.5 + offset_x, -3.3 + offset_y, 0.1), rot=(-0.3, 0, -0.8, 0), scale=1.15)
        actor11 = load_stone_asset(self._gym, self._sim, env=env, name="Mountain Rock11", fix_base_link=False,
                                   pos=(19 + offset_x, -4 + offset_y, 0.1), rot=(-0.3, 0, -0.8, 0), scale=1.15)
        actor12 = load_stone_asset(self._gym, self._sim, env=env, name="Mountain Rock12", fix_base_link=False,
                                   pos=(24.1 + offset_x, -3 + offset_y, 0.1), rot=(-0.2, 0.8, 0.8, 0), scale=1.75)

        self._actors.append(actor1)
        self._actors.append(actor2)
        self._actors.append(actor3)
        self._actors.append(actor4)
        self._actors.append(actor5)
        self._actors.append(actor6)
        self._actors.append(actor7)
        self._actors.append(actor8)
        self._actors.append(actor9)
        self._actors.append(actor10)
        self._actors.append(actor11)
        self._actors.append(actor12)

    def load_snow_rocks(self, env, offset_x, offset_y):
        # Big Snow Rocks
        actor1 = load_stone_asset(self._gym, self._sim, env=env, name="Snow Rock1", fix_base_link=False,
                                  pos=(12 + offset_x, -0.5 + offset_y, 0.1), rot=(0, 0, 1, 0),
                                  apply_texture=False, scale=0.25)
        actor2 = load_stone_asset(self._gym, self._sim, env=env, name="Snow Rock2", fix_base_link=False,
                                  pos=(14 + offset_x, offset_y, 0.1), rot=(0.3, 0.2, 0.1, 1),
                                  apply_texture=False, scale=0.3)
        actor3 = load_stone_asset(self._gym, self._sim, env=env, name="Snow Rock3", fix_base_link=False,
                                  pos=(17 + offset_x, -1 + offset_y, 0.1), rot=(0.2, 0.1, 0.8, 0),
                                  apply_texture=False, scale=0.35)
        actor4 = load_stone_asset(self._gym, self._sim, env=env, name="Snow Rock4", fix_base_link=False,
                                  pos=(10 + offset_x, offset_y, 0.1), rot=(0., 0., 1., 0.),
                                  apply_texture=False, scale=0.4)
        actor5 = load_stone_asset(self._gym, self._sim, env=env, name="Snow Rock5", fix_base_link=False,
                                  pos=(18 + offset_x, -8 + offset_y, 0.1), rot=(0, 0, 0, 1),
                                  apply_texture=False, scale=0.4)
        actor6 = load_stone_asset(self._gym, self._sim, env=env, name="Snow Rock6", fix_base_link=False,
                                  pos=(10 + offset_x, -7 + offset_y, 0.3), rot=(0.2, 0, 0.8, 0),
                                  apply_texture=False, scale=0.3)
        actor7 = load_stone_asset(self._gym, self._sim, env=env, name="Snow Rock7", fix_base_link=False,
                                  pos=(13 + offset_x, -8.5 + offset_y, 0.1), rot=(0.2, 0, 0.8, 0),
                                  apply_texture=False, scale=0.55)
        actor8 = load_stone_asset(self._gym, self._sim, env=env, name="Snow Rock8", fix_base_link=False,
                                  pos=(10 + offset_x, -3 + offset_y, 0.1), rot=(0.2, 0, 0.8, 0),
                                  apply_texture=False, scale=0.25)

        self._actors.append(actor1)
        self._actors.append(actor2)
        self._actors.append(actor3)
        self._actors.append(actor4)
        self._actors.append(actor5)
        self._actors.append(actor6)
        self._actors.append(actor7)
        self._actors.append(actor8)

    def load_random_snowstones_in_a_region(self, env, scene_offset_x=40, width=4.0, length=5.0, stone_nums=10,
                                           reverse=False):
        if reverse:
            offset_x = -13 + scene_offset_x
        else:
            offset_x = 8.1 + scene_offset_x
        offset_y = -5
        np.random.seed(25)
        for i in range(stone_nums):
            random_width = np.random.uniform(low=0, high=width)
            random_length = np.random.uniform(low=0, high=length)
            random_quat = random_quaternion()
            scale = np.random.uniform(low=0.1, high=0.18)

            x = offset_x + random_length
            y = offset_y + random_width

            stone_actor = load_stone_asset(self._gym, self._sim, env=env, name=f"snow_stones{i}", pos=(x, y, 0.03),
                                           fix_base_link=False, rot=tuple(random_quat), apply_texture=False,
                                           scale=scale)
            self._actors.append(stone_actor)

    def load_random_cobblestones_in_a_region(self, env, scene_offset_x=40, width=2.0, length=2.0, stone_nums=10,
                                             reverse=False):
        if reverse:
            offset_x = -8.1 + scene_offset_x
        else:
            offset_x = 6 + scene_offset_x
        offset_y = -0.985
        np.random.seed(12)
        for i in range(stone_nums):
            random_width = np.random.uniform(low=0, high=width)
            random_length = np.random.uniform(low=0, high=length)
            random_quat = random_quaternion()
            scale = np.random.uniform(low=0.06, high=0.08)

            x = offset_x + random_length
            y = offset_y + random_width

            stone_actor = load_stone_asset(self._gym, self._sim, env=env, pos=(x, y, 0.007), name=f"cobblestones{i}",
                                           fix_base_link=True, rot=tuple(random_quat), apply_texture=True, scale=scale)
            self._actors.append(stone_actor)

    def load_stop_sign(self, env, offset_x, offset_y):
        # Stop Sign
        actor1 = load_stop_sign_asset(self._gym, self._sim, env=env, name="Stop Sign 1", fix_base_link=True,
                                      disable_gravity=True, pos=(14 + offset_x, -0.5 + offset_y, 0.1),
                                      rot=(0.5, -0.5, -0.5, 0.5), scale=0.75)
        self._actors.append(actor1)

    def load_snow_tree(self, env, offset_x, offset_y):
        # Snow Tree
        actor1 = load_snow_tree_asset(self._gym, self._sim, env=env, name="Snow Tree 1", fix_base_link=True,
                                      pos=(8 + offset_x, -0.5 + offset_y, 0.4), rot=(0., 0., 0., 1.), scale=0.2)
        self._actors.append(actor1)

    def load_oak_tree(self, env, offset_x, offset_y):
        # Oak Tree
        actor1 = load_oak_tree_asset(self._gym, self._sim, env=env, name="Oak Tree 1", fix_base_link=True,
                                     pos=(15 + offset_x, -9 + offset_y, 0.), rot=(0.707, 0, 0, 0.707), scale=0.2)
        self._actors.append(actor1)

    def load_dead_wood(self, env, offset_x, offset_y):
        # Dead wood
        actor1 = load_dead_wood_asset(self._gym, self._sim, env=env, name="Dead Wood 1", fix_base_link=True,
                                      pos=(17 + offset_x, -1 + offset_y, -0.2), rot=(0.707, 0, 0, 0.707), scale=1.)
        self._actors.append(actor1)

    def check_collision(self, robot_env_handle, obstacle_id):
        """check collision between the robot and the obstacle"""

        # self._gym.

        contact_tensor = self._gym.get_sim_rigid_contacts(sim)
        if contact_tensor is None:
            return False

        contact_data = gymtorch.wrap_tensor(contact_tensor)

        for contact in contact_data:
            actor0 = contact[0].item()  # 获取 actor0 ID
            actor1 = contact[1].item()  # 获取 actor1 ID

            if (actor0 == robot_id and actor1 == obstacle_id) or (actor0 == obstacle_id and actor1 == robot_id):
                return True  # 发生碰撞

        return False  # 没有发生碰撞
