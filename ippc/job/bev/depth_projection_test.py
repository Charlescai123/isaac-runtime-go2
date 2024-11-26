import matplotlib.pyplot as plt
import numpy as np
import pybullet as pb
import pybullet_data

"""
The script is partly borrowed from https://gist.github.com/dmklee/a41f6ed11601967222d0120f620e7dcb
"""


def scale_to_255(a, min, max, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-255
        Optionally specify the data type of the output (default is uint8)
    """
    return (((a - min) / float(max - min)) * 255).astype(dtype)


def getDepth(z_n, zNear, zFar):
    z_n = 2.0 * z_n - 1.0
    z_e = 2.0 * zNear * zFar / (zFar + zNear - z_n * (zFar - zNear))
    return z_e


def birds_eye_point_cloud(points,
                          side_range=(-2, 2),
                          fwd_range=(0, 4),
                          res=0.01,
                          min_height=-1,
                          max_height=2,
                          as_occupancy=False):
    """ Creates an 2D birds eye view representation of the point cloud data.
        You can optionally save the image to specified filename.

    Args:
        points:     (numpy array)
                    N rows of points data
                    Each point should be specified by at least 3 elements x,y,z
        side_range: (tuple of two floats)
                    (-left, right) in metres
                    left and right limits of rectangle to look at.
        fwd_range:  (tuple of two floats)
                    (-behind, front) in metres
                    back and front limits of rectangle to look at.
        res:        (float) desired resolution in metres to use
                    Each output pixel will represent an square region res x res
                    in size.
        min_height:  (float)(default=-2.73)
                    Used to truncate height values to this minumum height
                    relative to the sensor (in metres).
                    The default is set to -2.73, which is 1 metre below a flat
                    road surface given the configuration in the kitti dataset.
        max_height: (float)(default=1.27)
                    Used to truncate height values to this maximum height
                    relative to the sensor (in metres).
                    The default is set to 1.27, which is 3m above a flat road
                    surface given the configuration in the kitti dataset.
        saveto:     (str or None)(default=None)
                    Filename to save the image as.
                    If None, then it just displays the image.
    """
    x_lidar = points[:, 0]
    y_lidar = points[:, 1]
    z_lidar = points[:, 2]

    ff = np.logical_and((x_lidar > fwd_range[0]), (x_lidar < fwd_range[1]))
    ss = np.logical_and((y_lidar > -side_range[1]), (y_lidar < -side_range[0]))
    indices = np.argwhere(np.logical_and(ff, ss)).flatten()

    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (-y_lidar[indices] / res).astype(np.int32)  # x axis is -y in LIDAR
    y_img = (x_lidar[indices] / res).astype(np.int32)  # y axis is -x in LIDAR
    # will be inverted later

    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor used to prevent issues with -ve vals rounding upwards
    x_img -= int(np.floor(side_range[0] / res))
    y_img -= int(np.floor(fwd_range[0] / res))

    # FILL PIXEL VALUES IN IMAGE ARRAY
    x_max = int((side_range[1] - side_range[0]) / res)
    y_max = int((fwd_range[1] - fwd_range[0]) / res)

    if as_occupancy:
        # initialize as unknown
        # mask unknown as -1
        # occupied as 1
        # free as 0
        im = -1 * np.ones([y_max, x_max], dtype=np.uint8)  # initialize grid as unknown (-1)
        height = z_lidar[indices]
        occupancy_threshold = 0.3
        height[height > occupancy_threshold] = 1
        height[height <= occupancy_threshold] = 0
        pixel_values = scale_to_255(height, min=-1, max=1)
        im[-y_img, x_img] = pixel_values
        print(im)
    else:
        # CLIP HEIGHT VALUES - to between min and max heights
        pixel_values = np.clip(a=z_lidar[indices],
                               a_min=min_height,
                               a_max=max_height)

        # RESCALE THE HEIGHT VALUES - to be between the range 0-255
        pixel_values = scale_to_255(pixel_values, min=min_height, max=max_height)
        im = np.zeros([y_max, x_max], dtype=np.uint8)
        im[-y_img, x_img] = pixel_values  # -y because images start from top left

    return im


def read_parameters(dbg_params):
    """Reads values from debug parameters

    Parameters
    ----------
    dbg_params : dict
        Dictionary where the keys are names (str) of parameters and the values are
        the itemUniqueId (int) for the corresponing debug item in pybullet

    Returns
    -------
    dict
        Dictionary that maps parameter names (str) to parameter values (float)
    """
    values = dict()
    for name, param in dbg_params.items():
        values[name] = pb.readUserDebugParameter(param)

    return values


def initialize_simulator():
    '''Creates a pybullet simulator in GUI mode and adds some objects so the
    camera has something to look at
    '''
    pb.connect(pb.GUI)
    pb.resetDebugVisualizerCamera(5, -90, -30, [0.0, -0.0, -0.0])
    pb.setTimeStep(1 / 240.)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    planeId = pb.loadURDF('plane.urdf')
    # textureId = pb.loadTexture("quad_gym/env/assets/grass.png")
    # pb.changeVisualShape(planeId, -1, textureUniqueId=textureId, flags=1)

    obj_initial_orientation_euler = np.array([0, 0, 0])

    scale = 2.5
    obj_initial_orientation = pb.getQuaternionFromEuler(obj_initial_orientation_euler)
    #
    # FLAG_TO_FILENAME = {
    #     'mounts': "quad_gym/env/assets/heightmaps/wm_height_out.png",
    #     'maze': "quad_gym/env/assets/heightmaps/Maze.png"
    # }
    # terrain_shape = pb.createCollisionShape(
    #     shapeType=pb.GEOM_HEIGHTFIELD,
    #     meshScale=[0.1, 0.1, 15 * 1.56],
    #     flags=1,
    #     fileName=FLAG_TO_FILENAME["mounts"])
    # terrain = pb.createMultiBody(0, terrain_shape)
    # textureId = pb.loadTexture(
    #     "quad_gym/env/assets/heightmaps/gimp_overlay_out.png")
    # pb.changeVisualShape(
    #     terrain, -1, textureUniqueId=textureId, flags=1)
    # # Move Origin A little bit to start at Flat Area
    # pb.resetBasePositionAndOrientation(
    #     terrain, [2, 2, 2.26], [0, 0, 0, 1])
    # pb.changeVisualShape(
    #     terrain, -1, rgbaColor=[1, 1, 1, 1])

    pb.loadURDF("quad_gym/env/assets/stone.urdf",
                basePosition=[0, 0, 0.1],
                baseOrientation=obj_initial_orientation,
                globalScaling=scale
                )

    pb.loadURDF("quad_gym/env/assets/stone.urdf",
                basePosition=[0, -1.0, 0.1],
                baseOrientation=obj_initial_orientation,
                globalScaling=scale * 0.8
                )

    pb.loadURDF("quad_gym/env/assets/stone.urdf",
                basePosition=[-1.5, 0.5, 0.1],
                baseOrientation=obj_initial_orientation,
                globalScaling=scale * 0.5
                )

    pb.loadURDF("quad_gym/env/assets/stone.urdf",
                basePosition=[-2.0, -0.9, 0.1],
                baseOrientation=obj_initial_orientation,
                globalScaling=scale * 0.3
                )

    pb.setGravity(0, 0, -9.8)
    pb.setRealTimeSimulation(1)


def interactive_camera_placement(pos_scale=1.,
                                 max_dist=6.,
                                 show_plot=True,
                                 verbose=True,
                                 ):
    """GUI for adjusting camera placement in pybullet. Use the scales to adjust
    intuitive parameters that govern view and projection matrix.  When you are
    satisfied, you can hit the print button and the values needed to recreate
    the camera placement will be logged to console.
    In addition to showing a live feed of the camera, there are also two visual
    aids placed in the simulator to help understand camera placement: the target
    position and the camera. These are both shown as red objects and can be
    viewed using the standard controls provided by the GUI.
    Note
    ----
    There must be a simulator running in GUI mode for this to work
    Parameters
    ----------
    pos_scale : float
        Position scaling that limits the target position of the camera.
    max_dist : float
        Maximum distance the camera can be away from the target position, you
        may need to adjust if you scene is large
    show_plot : bool, default to True
        If True, then a matplotlib window will be used to plot the generated
        image.  This is beneficial if you want different values for image width
        and height (since the built in image visualizer in pybullet is always
        square).
    verbose : bool, default to False
        If True, then additional parameters will be printed when print button
        is pressed.
    """
    np.set_printoptions(suppress=True, precision=4)

    dbg = dict()
    # for view matrix
    # dbg['target_x'] = pb.addUserDebugParameter('target_x', -pos_scale, pos_scale, 0)
    # dbg['target_y'] = pb.addUserDebugParameter('target_y', -pos_scale, pos_scale, 0)
    # dbg['target_z'] = pb.addUserDebugParameter('target_z', -pos_scale, pos_scale, 0)
    # dbg['distance'] = pb.addUserDebugParameter('distance', 0, max_dist, max_dist / 2)
    # dbg['yaw'] = pb.addUserDebugParameter('yaw', -180, 180, 0)
    # dbg['pitch'] = pb.addUserDebugParameter('pitch', -180, 180, -90)
    # dbg['roll'] = pb.addUserDebugParameter('roll', -180, 180, 0)


    dbg['target_x'] = pb.addUserDebugParameter('target_x', -pos_scale, pos_scale, 0)
    dbg['target_y'] = pb.addUserDebugParameter('target_y', -pos_scale, pos_scale, 0)
    dbg['target_z'] = pb.addUserDebugParameter('target_z', -pos_scale, pos_scale, 0)

    dbg['pose_x'] = pb.addUserDebugParameter('pose_x', -5, 5, 0)
    dbg['pose_y'] = pb.addUserDebugParameter('pose_y', -5, 5, 0)
    dbg['pose_z'] = pb.addUserDebugParameter('pose_z',  0, 5, 1)

    dbg['up_x'] = pb.addUserDebugParameter('up_x',  0, 1, 0)
    dbg['up_y'] = pb.addUserDebugParameter('up_y',  0, 1, 0)
    dbg['up_z'] = pb.addUserDebugParameter('up_z',  0, 1, 1)

    dbg['upAxisIndex'] = pb.addUserDebugParameter('toggle upAxisIndex', 1, 0, 1)

    # for projection matrix
    dbg['width'] = pb.addUserDebugParameter('width', 100, 1000, 240)
    dbg['height'] = pb.addUserDebugParameter('height', 100, 1000, 240)
    dbg['fov'] = pb.addUserDebugParameter('fov', 1, 180, 90)
    dbg['near_val'] = pb.addUserDebugParameter('near_val', 1e-6, 1, 0.1)
    dbg['far_val'] = pb.addUserDebugParameter('far_val', 1, 10, 5)

    # visual aids for target and camera pose
    target_vis_id = pb.createVisualShape(pb.GEOM_SPHERE,
                                         radius=0.1,
                                         rgbaColor=[1, 0, 0, 0.7])
    target_body = pb.createMultiBody(0, -1, target_vis_id)

    camera_vis_id = pb.createVisualShape(pb.GEOM_BOX,
                                         halfExtents=[0.02, 0.05, 0.02],
                                         rgbaColor=[1, 0, 0, 0.7])
    camera_body = pb.createMultiBody(0, -1, camera_vis_id)

    # pyplot window to show feed
    if show_plot:
        f, axarr = plt.subplots(2, 2, figsize=(16, 16))
        plt.axis('off')
        plt.tight_layout(pad=0)

    # Start simulation
    dbg['print'] = pb.addUserDebugParameter('print params', 1, 0, 1)
    old_print_val = 1

    while 1:
        values = read_parameters(dbg)

        # target_pos = np.array([values[f'target_{c}'] for c in 'xyz'])
        upAxisIndex = (int(values['upAxisIndex']) % 2) + 1
        # view_mtx = pb.computeViewMatrixFromYawPitchRoll(target_pos,
        #                                                 values['distance'],
        #                                                 values['yaw'],
        #                                                 values['pitch'],
        #                                                 values['roll'],
        #                                                 upAxisIndex)

        # Define a camera view matrix
        camPos = [values['pose_x'], values['pose_y'], values['pose_z']]
        camTarget = [values['target_x'], values['target_y'], values['target_z']]
        camUp = [values['up_x'], values['up_y'], values['up_z']]

        view_mtx = pb.computeViewMatrix(
            cameraEyePosition=camPos,
            cameraTargetPosition=camTarget,
            cameraUpVector=camUp)

        width = int(values['width'])
        height = int(values['height'])
        aspect = width / height
        proj_mtx = pb.computeProjectionMatrixFOV(values['fov'],
                                                 aspect,
                                                 values['near_val'],
                                                 values['far_val'])

        # update visual aids for camera, target pos
        pb.resetBasePositionAndOrientation(target_body, camTarget, [0, 0, 0, 1])

        view_mtx = np.array(view_mtx).reshape((4, 4), order='F')
        # cam_pos = np.dot(view_mtx[:3, :3].T, -view_mtx[:3, 3])
        # cam_euler = np.radians([values['pitch'], values['roll'], values['yaw']])

        cam_quat = pb.getQuaternionFromEuler(camUp)
        pb.resetBasePositionAndOrientation(camera_body, camPos, cam_quat)

        view_mtx = view_mtx.reshape(-1, order='F')

        imgW, imgH, rgbImg, depthImg, segImg = pb.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_mtx,
            projectionMatrix=proj_mtx,
            renderer=pb.ER_BULLET_HARDWARE_OPENGL
        )

        depth = pb.getCameraImage(width, height, view_mtx, proj_mtx)[3]

        stepX = 1
        stepY = 1
        realDepthImg = depth.copy()

        for w in range(0, imgW, stepX):
            for h in range(0, imgH, stepY):
                realDepthImg[w][h] = getDepth(depthImg[w][h], values['near_val'], values['far_val'])

        pointCloud = np.empty([np.int32(imgH / stepY), np.int32(imgW / stepX), 4])

        projectionMatrix = np.asarray(proj_mtx).reshape([4, 4], order='F')

        viewMatrix = np.asarray(view_mtx).reshape([4, 4], order='F')

        tran_pix_world = np.linalg.inv(np.matmul(projectionMatrix, viewMatrix))

        for h in range(0, imgH, stepY):
            for w in range(0, imgW, stepX):
                x = (2 * w - imgW) / imgW
                y = -(2 * h - imgH) / imgH  # be careful！ deepth and its corresponding position
                z = 2 * depthImg[h, w] - 1
                # z = realDepthImg[h,w]
                pixPos = np.asarray([x, y, z, 1])
                # print(pixPos)
                position = np.matmul(tran_pix_world, pixPos)
                pointCloud[np.int32(h / stepY), np.int32(w / stepX), :] = position / position[3]

        pointCloud = np.reshape(pointCloud[:, :, :3], newshape=(-1, 3))
        # center_point_cloud_around_camera
        pointCloud -= np.array([camPos])
        pointCloud[:, -1] *= -1
        bev_img = birds_eye_point_cloud(pointCloud, as_occupancy=True)

        if show_plot:
            axarr[0, 0].imshow(rgbImg)
            axarr[0, 1].imshow(realDepthImg)
            axarr[1, 0].imshow(segImg)
            axarr[1, 1].imshow(bev_img)
            plt.pause(0.1)

        if old_print_val != values['print']:
            old_print_val = values['print']
            print("\n========================================")
            print(f"VIEW MATRIX : \n{np.array_str(view_mtx)}")
            print(f"PROJECTION MATRIX : \n{np.array_str(view_mtx)}")
            if verbose:
                print(f"target position : {np.array_str(camTarget)}")
                print(f"distance : {dbg['distance']:.2f}")
                print(f"yaw : {dbg['yaw']:.2f}")
                print(f"pitch : {dbg['pitch']:.2f}")
                print(f"roll : {dbg['roll']:.2f}")
                print(f"upAxisIndex : {upAxisIndex:d}")
                print(f"width : {width:d}")
                print(f"height : {height:d}")
                print(f"fov : {dbg['fov']:.1f}")
                print(f"aspect : {aspect:.2f}")
                print(f"nearVal : {dbg['near_val']:.2f}")
                print(f"farVal : {dbg['far_val']:.2f}")
            print("========================================\n")


if __name__ == "__main__":
    # Substitute this for the initialization of your simulator
    initialize_simulator()
    interactive_camera_placement()
