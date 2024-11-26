"""
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Depth Camera to Point Cloud Exmaple
-----------------------------------
An example which shows how to deproject the depth camera ground truth image
from gym into a 3D point cloud.

Requires pptk toolkit for viewing the resulting point cloud (pip install pptk)

Note: If pptk viewer stalls on Ubuntu, refer to https://github.com/heremaps/pptk/issues/3 (remove libz from package so it uses system libz)

"""

import numpy as np
# import pptk
from isaacgym import gymutil
from isaacgym import gymapi

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(description="Depth Camera To Point Cloud Example", headless=True)

# configure sim
sim_params = gymapi.SimParams()
sim_params.dt = 1.0 / 60.0
if args.physics_engine == gymapi.SIM_FLEX:
    sim_params.flex.solver_type = 5
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 15
    sim_params.flex.relaxation = 0.75
    sim_params.flex.warm_start = 0.4
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()

# create viewer
if not args.headless:
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise ValueError('*** Failed to create viewer')

# add ground plane with segmentation id zero
# so we can identify depths originating from the ground
# plane and ignore them
plane_params = gymapi.PlaneParams()
plane_params.segmentation_id = 0
gym.add_ground(sim, plane_params)

# set up the env grid parameters
num_envs = 1
spacing = 1
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# Create a Sphere
asset_sphere_low = gym.create_sphere(sim, 0.2, gymapi.AssetOptions())

# Load assets
asset_root = "../../assets"

# Load the Franka Arm
load_options = gymapi.AssetOptions()
load_options.fix_base_link = True
load_options.flip_visual_attachments = True
load_options.disable_gravity = True
load_options.armature = 0.01
franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"
franka_asset = gym.load_asset(sim, asset_root, franka_asset_file, load_options)

# Load the Sektion cabinet
load_options.flip_visual_attachments = False
load_options.disable_gravity = False
cabinet_asset_file = "urdf/sektion_cabinet_model/urdf/sektion_cabinet.urdf"
cabinet_asset = gym.load_asset(sim, asset_root, cabinet_asset_file, load_options)

envs = []


for i in range(num_envs):
    # create environment
    env = gym.create_env(sim, env_lower, env_upper, 8)
    envs.append(env)
    q = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
    # Place 3 actors in the environment
    gym.create_actor(env, asset_sphere_low, gymapi.Transform(r=q, p=gymapi.Vec3(0.0, 0.25, 0.0)), 'box', i, 1, segmentationId=10)
    gym.create_actor(env, franka_asset, gymapi.Transform(r=q, p=gymapi.Vec3(2.0, 0.0, 0.0)), 'franka', i, 1, segmentationId=11)
    gym.create_actor(env, cabinet_asset, gymapi.Transform(r=q, p=gymapi.Vec3(0.0, 0.4, 2.0)), 'cab', i, 1, segmentationId=12)

# Camera properties
cam_positions = []
cam_targets = []
cam_handles = []
cam_width = 480
cam_height = 320
cam_props = gymapi.CameraProperties()
cam_props.width = cam_width
cam_props.height = cam_height

# Camera 0 Position and Target
cam_positions.append(gymapi.Vec3(2, 0.5, 1.5))
cam_targets.append(gymapi.Vec3(0.0, 0.5, 0.0))

# Camera 1 Position and Target
cam_positions.append(gymapi.Vec3(-0.5, 0.5, 2))
cam_targets.append(gymapi.Vec3(0.0, 0.5, 0.0))

# Camera 2 Position and Target
cam_positions.append(gymapi.Vec3(2.333, 2.5, -2))
cam_targets.append(gymapi.Vec3(0.0, 0.5, 0.0))

# Camera 3 Position and Target
cam_positions.append(gymapi.Vec3(2.2, 1.5, -2))
cam_targets.append(gymapi.Vec3(0.0, 0.5, 0.0))

# Camera 4 Position and Target
cam_positions.append(gymapi.Vec3(2, 2.5, -0.5))
cam_targets.append(gymapi.Vec3(0.0, 0.5, 0.0))

# Create cameras in environment zero and set their locations
# to the above
env = envs[0]
for c in range(len(cam_positions)):
    cam_handles.append(gym.create_camera_sensor(env, cam_props))
    gym.set_camera_location(cam_handles[c], env, cam_positions[c], cam_targets[c])


if not args.headless:
    gym.viewer_camera_look_at(viewer, envs[0], gymapi.Vec3(3, 2, 3), gymapi.Vec3(0, 0, 0))

frame_count = 0

while True:
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update graphics
    gym.step_graphics(sim)

    # Update viewer and check for exit conditions
    if not args.headless:
        if gym.query_viewer_has_closed(viewer):
            break
        gym.draw_viewer(viewer, sim, False)

    # deprojection is expensive, so do it only once on the 2nd frame
    if frame_count == 1:
        # Array of RGB Colors, one per camera, for dots in the resulting
        # point cloud. Points will have a color which indicates which camera's
        # depth image created the point.
        color_map = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 1], [1, 0, 1]])

        # Render all of the image sensors only when we need their output here
        # rather than every frame.
        gym.render_all_camera_sensors(sim)

        points = []
        color = []
        print("Converting Depth images to point clouds. Have patience...")
        for c in range(len(cam_handles)):
            print("Deprojecting from camera %d" % c)
            # Retrieve depth and segmentation buffer
            depth_buffer = gym.get_camera_image(sim, env, cam_handles[c], gymapi.IMAGE_DEPTH)
            seg_buffer = gym.get_camera_image(sim, env, cam_handles[c], gymapi.IMAGE_SEGMENTATION)

            # Get the camera view matrix and invert it to transform points from camera to world
            # space
            vinv = np.linalg.inv(np.matrix(gym.get_camera_view_matrix(sim, env, cam_handles[c])))

            # Get the camera projection matrix and get the necessary scaling
            # coefficients for deprojection
            proj = gym.get_camera_proj_matrix(sim, env, cam_handles[c])
            fu = 2/proj[0, 0]
            fv = 2/proj[1, 1]

            # Ignore any points which originate from ground plane or empty space
            depth_buffer[seg_buffer == 0] = -10001

            centerU = cam_width/2
            centerV = cam_height/2
            for i in range(cam_width):
                for j in range(cam_height):
                    if depth_buffer[j, i] < -10000:
                        continue
                    if seg_buffer[j, i] > 0:
                        u = -(i-centerU)/(cam_width)  # image-space coordinate
                        v = (j-centerV)/(cam_height)  # image-space coordinate
                        d = depth_buffer[j, i]  # depth buffer value
                        X2 = [d*fu*u, d*fv*v, d, 1]  # deprojection vector
                        p2 = X2*vinv  # Inverse camera view to get world coordinates
                        points.append([p2[0, 2], p2[0, 0], p2[0, 1]])
                        color.append(c)

        # # use pptk to visualize the 3d point cloud created above
        # v = pptk.viewer(points, color)
        # v.color_map(color_map)
        # # Sets a similar view to the gym viewer in the PPTK viewer
        # v.set(lookat=[0, 0, 0], r=5, theta=0.4, phi=0.707)
        # print("Point Cloud Complete")
        import open3d as o3d

        # Convert points and color to numpy arrays
        points = np.array(points)
        print(f"points: {points}")
        print(f"points: {points.shape}")
        colors = np.array([color_map[c] for c in color])

        print(f"colors: {colors}")
        print(f"colors: {colors.shape}")

        # Create Open3D PointCloud object
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

        # Visualize the point cloud
        o3d.visualization.draw_geometries([point_cloud],
                                          zoom=0.8,
                                          front=[0.0, 0.0, -1.0],
                                          lookat=[0.0, 0.0, 0.0],
                                          up=[0.0, -1.0, 0.0])
        print("Point Cloud Visualization Complete")

        # In headless mode, quit after the deprojection is complete
        # The pptk viewer will remain valid until its window is closed
        if args.headless:
            break

    frame_count = frame_count + 1

if not args.headless:
    gym.destroy_viewer(viewer)

gym.destroy_sim(sim)