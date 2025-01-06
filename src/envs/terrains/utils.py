import warnings
import numpy as np
from isaacgym import gymtorch, gymapi
from isaacgym.terrain_utils import *

ASSET_ROOT = "resources/terrains"

STONE_ASSET = "stone.urdf"
CEMENT_ASSET = "plane/cement.urdf"


def random_quaternion():
    u1, u2, u3 = np.random.uniform(0.0, 1.0, 3)

    w = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    x = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    y = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    z = np.sqrt(u1) * np.cos(2 * np.pi * u3)

    return np.array([w, x, y, z])


def load_cement_road_asset(gym, sim, env, name="cement_road", collision_group=-1, collision_filter=-1, pos=(0, 0, 0),
                           rot=(0, 0, 0, 1), apply_texture=True, scale=1.):
    x, y, z = pos[:]
    X, Y, Z, W = rot[:]
    transform = gymapi.Transform(p=gymapi.Vec3(x, y, z), r=gymapi.Quat(X, Y, Z, W))

    asset_root = ASSET_ROOT
    asset_urdf = CEMENT_ASSET
    asset_options = gymapi.AssetOptions()

    # Load materials from meshes
    asset_options.use_mesh_materials = apply_texture
    asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX

    # Override the bogus inertia tensors and center-of-mass properties in the YCB assets.
    # These flags will force the inertial properties to be recomputed from geometry.
    asset_options.override_inertia = False
    asset_options.override_com = False
    asset_options.fix_base_link = True
    asset_options.disable_gravity = False

    # use default convex decomposition params
    # asset_options.vhacd_enabled = True
    asset_cement_road = gym.load_asset(sim, asset_root, asset_urdf, asset_options)

    # create actor
    actor_cement_road = gym.create_actor(env, asset_cement_road, transform, name, collision_group, collision_filter)
    scale_status = gym.set_actor_scale(env, actor_cement_road, scale)
    # scale_status = False
    # if scale_status:
    #     print(f"Set cement road actor scale successfully")
    # else:
    #     warnings.warn(f"Failed to set the actor scale")

    return actor_cement_road


def load_stone_asset(gym, sim, env, name="stone", pos=(0, 0, 0), rot=(0, 0, 0, 1), scale=1., collision_group=-1,
                     collision_filter=-1, fix_base_link=False, apply_texture=True, override_inertia=False,
                     override_com=False, disable_gravity=False):
    x, y, z = pos[:]
    X, Y, Z, W = rot[:]
    transform = gymapi.Transform(p=gymapi.Vec3(x, y, z), r=gymapi.Quat(X, Y, Z, W))

    asset_root = ASSET_ROOT
    asset_urdf = STONE_ASSET
    asset_options = gymapi.AssetOptions()

    # Load materials from meshes
    asset_options.default_dof_drive_mode = 0
    asset_options.use_mesh_materials = apply_texture
    asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX

    # Override the bogus inertia tensors and center-of-mass properties in the YCB assets.
    # These flags will force the inertial properties to be recomputed from geometry.
    asset_options.override_inertia = override_inertia
    asset_options.override_com = override_com
    asset_options.fix_base_link = fix_base_link
    asset_options.disable_gravity = disable_gravity

    # use default convex decomposition params
    # asset_options.vhacd_enabled = True
    asset_stone = gym.load_asset(sim, asset_root, asset_urdf, asset_options)

    # create actor
    actor_stone = gym.create_actor(env, asset_stone, transform, name, collision_group, collision_filter)
    scale_status = gym.set_actor_scale(env, actor_stone, scale)
    # scale_status = False
    # if scale_status:
    #     print(f"Set stone actor scale successfully")
    # else:
    #     warnings.warn(f"Failed to set the actor scale")

    return actor_stone


def load_random_snowstones_in_a_region(gym, sim, env, scene_offset_x=40, width=4.0, length=5.0, stone_nums=10,
                                       reverse=False):
    if reverse:
        offset_x = -13 + scene_offset_x
    else:
        offset_x = 8.1 + scene_offset_x
    offset_y = -1.8
    np.random.seed(25)
    for i in range(stone_nums):
        random_width = np.random.uniform(low=0, high=width)
        random_length = np.random.uniform(low=0, high=length)
        random_quat = random_quaternion()
        scale = np.random.uniform(low=0.1, high=0.18)

        x = offset_x + random_length
        y = offset_y + random_width

        load_stone_asset(gym=gym, sim=sim, env=env, name=f"snow_stones{i}", pos=(x, y, 0.03), fix_base_link=False,
                         rot=tuple(random_quat), apply_texture=False, scale=scale)


def load_random_cobblestones_in_a_region(gym, sim, env, scene_offset_x=40, width=2.0, length=2.0, stone_nums=10,
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

        load_stone_asset(gym=gym, sim=sim, env=env, pos=(x, y, 0.007), name=f"cobblestones{i}", fix_base_link=True,
                         rot=tuple(random_quat), apply_texture=True, scale=scale)


def add_uneven_terrains(gym, sim, scene_offset_x=40, reverse=False):
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
    tm_params.transform.p.y = -terrain_width / 2 - 1. + 1
    gym.add_triangle_mesh(sim, vertices.flatten(), triangles.flatten(), tm_params)

    # vertices, triangles = convert_heightfield_to_trimesh(terrain.height_field_raw,
    #                                                      horizontal_scale=horizontal_scale,
    #                                                      vertical_scale=vertical_scale, slope_threshold=1.5)
    # gym.add_triangle_mesh(sim, vertices.flatten(), triangles.flatten(), tm_params)


def add_snow_road(self):
    env_lower = gymapi.Vec3(-5.0, -5.0, -5.0)
    env_upper = gymapi.Vec3(5.0, 5.0, 5.0)
    env = self._gym.create_env(self._sim, env_lower, env_upper, 1)

    # 添加一个平面作为地面
    # plane_params = gymapi.PlaneParams()
    # self._gym.add_ground(self._sim, plane_params)

    # 加载雪地纹理
    texture_path = "./meshes/normal_road.png"  # 替换为你的雪地纹理文件路径
    snow_texture_handle = self._gym.create_texture_from_file(self._sim, texture_path)

    # 创建地面几何体（一个大盒子来代表雪地）
    ground_dims = gymapi.Vec3(10.0, 10.0, 0.001)  # 地面的尺寸
    ground_geom = self._gym.create_box(self._sim, ground_dims.x, ground_dims.y, ground_dims.z)

    # 设置地面的变换
    ground_transform = gymapi.Transform()
    ground_transform.p = gymapi.Vec3(0.0, 0.0, ground_dims.z / 2)  # 确保地面位于正确位置

    # 创建地面 actor
    ground_actor = self._gym.create_actor(env, ground_geom, ground_transform, "snow_ground", 0, 0)

    # rigid_shape_props = self._gym.get_actor_rigid_shape_properties(env, ground_actor)
    # print(f"rigid: {rigid_shape_props[0]}")
    # rigid_shape_props[0].friction = 1
    # self._gym.set_actor_rigid_shape_properties(env, ground_actor, rigid_shape_props)

    # 将雪地纹理应用到地面 actor
    self._gym.set_rigid_body_texture(env, ground_actor, 0, gymapi.MESH_VISUAL, snow_texture_handle)


def arrow_plot(self, robot_pos=None):
    env = self._gym.create_env(self._sim, gymapi.Vec3(-1, -1, -1), gymapi.Vec3(1, 1, 1), 1)

    x, y, z = robot_pos

    # 定义箭头的顶点
    # 箭头的主轴（例如，Z 轴方向箭头）
    # arrow_start = np.array([0, 0, 0], dtype=np.float32)
    # arrow_end = np.array([0, 0, 1], dtype=np.float32)  # 箭头指向的方向
    arrow_start = np.array([x, y, z + 2], dtype=np.float32)
    arrow_end = np.array([0, 0, 1], dtype=np.float32)  # 箭头指向的方向

    # 箭头的两个翅膀，形成箭头的头部
    arrow_wing1 = np.array([0.1, 0.0, 0.9], dtype=np.float32)
    arrow_wing2 = np.array([-0.1, 0.0, 0.9], dtype=np.float32)

    # 合并顶点，用于绘制线条
    line_vertices = np.array([
        arrow_start, arrow_end,  # 主轴
        arrow_end, arrow_wing1,  # 一个翅膀
        arrow_end, arrow_wing2  # 另一个翅膀
    ], dtype=np.float32)

    # 定义每条线的颜色（例如，红色箭头）
    colors = np.array([
        [1, 0, 0],  # 主轴颜色
        [1, 0, 0],  # 主轴颜色
        [1, 0, 0],  # 翅膀颜色
        [1, 0, 0],  # 翅膀颜色
    ], dtype=np.float32)

    # 绘制箭头
    num_lines = len(line_vertices) // 2
    self._gym.add_lines(self._viewer, env, num_lines, line_vertices, colors)
