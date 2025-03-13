import time
import warnings
import numpy as np
from isaacgym import gymtorch, gymapi
from isaacgym.terrain_utils import *

from src.envs.terrains.utils.sim_utils import add_ground, add_terrain

ASSET_ROOT = "resources/terrains"

STONE_ASSET = "objects/stone/stone.urdf"
STOPSIGN_ASSET = "objects/stopsign/stopsign.urdf"
SNOW_TREE_ASSET = "tree/snow_tree/snow_tree.urdf"
OAK_TREE_ASSET = "tree/oak_tree/oak_tree.urdf"
DEAD_WOOD_ASSET = "tree/deadwood/deadwood.urdf"

CEMENT_ROAD_ASSET = "plane/cement_road/cement_road.urdf"


def random_quaternion():
    u1, u2, u3 = np.random.uniform(0.0, 1.0, 3)

    w = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    x = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    y = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    z = np.sqrt(u1) * np.cos(2 * np.pi * u3)

    return np.array([w, x, y, z])


def load_asset(gym, sim, env, asset_urdf, name, pos=(0, 0, 0), rot=(0, 0, 0, 1), scale=1.,
               collision_group=-1, collision_filter=-1, fix_base_link=False, apply_texture=True,
               override_inertia=False, override_com=False, disable_gravity=False):
    """
    Generic function to load assets in Isaac Gym.

    Parameters:
        gym (GymAPI): Isaac Gym API handle
        sim (SimAPI): Simulation environment
        env (EnvAPI): The environment where the asset will be loaded
        asset_urdf (str): Path to the asset URDF file
        name (str): Name of the asset
        pos (tuple): Asset's world coordinates (x, y, z)
        rot (tuple): Asset's rotation (quaternion) (x, y, z, w)
        scale (float): Scale factor of the asset
        collision_group (int): Collision group ID
        collision_filter (int): Collision filter ID
        fix_base_link (bool): Whether to fix the base link
        apply_texture (bool): Whether to apply texture materials
        override_inertia (bool): Whether to override inertia
        override_com (bool): Whether to override the center of mass
        disable_gravity (bool): Whether to disable gravity
    """
    # Set the transform
    transform = gymapi.Transform(
        p=gymapi.Vec3(*pos),
        r=gymapi.Quat(*rot)
    )

    # Set asset options
    asset_options = gymapi.AssetOptions()
    asset_options.use_mesh_materials = apply_texture
    asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
    asset_options.override_inertia = override_inertia
    asset_options.override_com = override_com
    asset_options.fix_base_link = fix_base_link
    asset_options.disable_gravity = disable_gravity

    # Load the asset
    asset = gym.load_asset(sim, ASSET_ROOT, asset_urdf, asset_options)

    # Create actor
    actor = gym.create_actor(env, asset, transform, name, collision_group, collision_filter)

    # Set scaling
    scale_status = gym.set_actor_scale(env, actor, scale)
    if not scale_status:
        warnings.warn(f"Failed to set scale for {name}")

    return actor


def load_cement_road_asset(gym, sim, env, **kwargs):
    return load_asset(gym, sim, env, CEMENT_ROAD_ASSET, **kwargs)


def load_stone_asset(gym, sim, env, **kwargs):
    return load_asset(gym, sim, env, STONE_ASSET, **kwargs)


def load_dead_wood_asset(gym, sim, env, **kwargs):
    return load_asset(gym, sim, env, DEAD_WOOD_ASSET, **kwargs)


def load_oak_tree_asset(gym, sim, env, **kwargs):
    return load_asset(gym, sim, env, OAK_TREE_ASSET, **kwargs)


def load_snow_tree_asset(gym, sim, env, **kwargs):
    return load_asset(gym, sim, env, SNOW_TREE_ASSET, **kwargs)


def load_stop_sign_asset(gym, sim, env, **kwargs):
    return load_asset(gym, sim, env, STOPSIGN_ASSET, **kwargs)


def add_uneven_terrains(gym, sim, scene_offset_x=0, scene_offset_y=0, reverse=False):
    # terrains
    num_terrains = 1
    terrain_width = 12.
    terrain_length = 12.
    horizontal_scale = 0.1  # [m] resolution in x
    vertical_scale = 0.01  # [m] resolution in z
    num_rows = int(terrain_width / horizontal_scale)
    num_cols = int(terrain_length / horizontal_scale)
    heightfield = np.zeros((num_terrains * num_rows, num_cols), dtype=np.int16)

    def new_sub_terrain():
        return SubTerrain(width=num_rows, length=num_cols, vertical_scale=vertical_scale,
                          horizontal_scale=horizontal_scale)

    # Pit height and width
    # pit_depth = [0.1, 0.1]
    # from isaacgym import terrain_utils
    # terrain = terrain_utils.SubTerrain()
    # terrain.height_field_raw[:] = -round(np.random.uniform(pit_depth[0], pit_depth[1]) / terrain.vertical_scale)

    # np.random.seed(42)  # works for vel 0.3 m/s
    np.random.seed(3)  # works for all vel
    heightfield[0:1 * num_rows, :] = random_uniform_terrain(new_sub_terrain(), min_height=-0.01, max_height=0.01,
                                                            step=0.05, downsampled_scale=0.1).height_field_raw
    # heightfield[1 * num_rows:2 * num_rows, :] = sloped_terrain(new_sub_terrain(), slope=-0.5).height_field_raw
    # heightfield[2 * num_rows:3 * num_rows, :] = stairs_terrain(new_sub_terrain(), step_width=0.75,
    #                                                            step_height=-0.35).height_field_raw
    # heightfield[2 * num_rows:3 * num_rows, :] = heightfield[2 * num_rows:3 * num_rows, :][::-1]
    # heightfield[3 * num_rows:4 * num_rows, :] = pyramid_stairs_terrain(new_sub_terrain(), step_width=0.75,
    #                                                                    step_height=-0.5).height_field_raw

    # add the terrain as a triangle mesh
    vertices, triangles = convert_heightfield_to_trimesh(heightfield, horizontal_scale=horizontal_scale,
                                                         vertical_scale=vertical_scale, slope_threshold=1.5)
    tm_params = gymapi.TriangleMeshParams()
    tm_params.nb_vertices = vertices.shape[0]
    tm_params.nb_triangles = triangles.shape[0]
    if reverse:
        tm_params.transform.p.x = -19.8 + scene_offset_x
    else:
        tm_params.transform.p.x = 8. + scene_offset_x
    tm_params.transform.p.y = -terrain_width / 2 - 1. + 1 + scene_offset_y
    gym.add_triangle_mesh(sim, vertices.flatten(), triangles.flatten(), tm_params)

    # vertices, triangles = convert_heightfield_to_trimesh(terrain.height_field_raw,
    #                                                      horizontal_scale=horizontal_scale,
    #                                                      vertical_scale=vertical_scale, slope_threshold=1.5)
    # gym.add_triangle_mesh(sim, vertices.flatten(), triangles.flatten(), tm_params)


def add_snow_road(self):
    env_lower = gymapi.Vec3(-5.0, -5.0, -5.0)
    env_upper = gymapi.Vec3(5.0, 5.0, 5.0)
    env = self._gym.create_env(self._sim, env_lower, env_upper, 1)

    # plane_params = gymapi.PlaneParams()
    # self._gym.add_ground(self._sim, plane_params)

    texture_path = "./meshes/normal_road.png"
    snow_texture_handle = self._gym.create_texture_from_file(self._sim, texture_path)

    ground_dims = gymapi.Vec3(10.0, 10.0, 0.001)
    ground_geom = self._gym.create_box(self._sim, ground_dims.x, ground_dims.y, ground_dims.z)

    ground_transform = gymapi.Transform()
    ground_transform.p = gymapi.Vec3(0.0, 0.0, ground_dims.z / 2)

    ground_actor = self._gym.create_actor(env, ground_geom, ground_transform, "snow_ground", 0, 0)

    # rigid_shape_props = self._gym.get_actor_rigid_shape_properties(env, ground_actor)
    # print(f"rigid: {rigid_shape_props[0]}")
    # rigid_shape_props[0].friction = 1
    # self._gym.set_actor_rigid_shape_properties(env, ground_actor, rigid_shape_props)

    self._gym.set_rigid_body_texture(env, ground_actor, 0, gymapi.MESH_VISUAL, snow_texture_handle)


def arrow_plot(self, robot_pos=None):
    env = self._gym.create_env(self._sim, gymapi.Vec3(-1, -1, -1), gymapi.Vec3(1, 1, 1), 1)

    x, y, z = robot_pos

    # arrow_start = np.array([0, 0, 0], dtype=np.float32)
    # arrow_end = np.array([0, 0, 1], dtype=np.float32)
    arrow_start = np.array([x, y, z + 2], dtype=np.float32)
    arrow_end = np.array([0, 0, 1], dtype=np.float32)

    arrow_wing1 = np.array([0.1, 0.0, 0.9], dtype=np.float32)
    arrow_wing2 = np.array([-0.1, 0.0, 0.9], dtype=np.float32)

    line_vertices = np.array([
        arrow_start, arrow_end,
        arrow_end, arrow_wing1,
        arrow_end, arrow_wing2
    ], dtype=np.float32)

    colors = np.array([
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
    ], dtype=np.float32)

    num_lines = len(line_vertices) // 2
    self._gym.add_lines(self._viewer, env, num_lines, line_vertices, colors)


def create_slope_terrain(gym, sim):
    """Creation of a slope terrain (e.g., stairs)"""
    add_ground(gym, sim)
    add_terrain(gym, sim, "stair")
    add_terrain(gym, sim, "slope")
    add_terrain(gym, sim, "stair", 3.95, True)
    add_terrain(gym, sim, "stair", 0., True)