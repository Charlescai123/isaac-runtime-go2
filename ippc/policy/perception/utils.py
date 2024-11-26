import numpy as np
import matplotlib.pyplot as plt


def getDepth(z_n, zNear, zFar):
    z_n = 2.0 * z_n - 1.0
    z_e = 2.0 * zNear * zFar / (zFar + zNear - z_n * (zFar - zNear))
    return z_e


def scale_to_255(a, min, max, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-255
        Optionally specify the data type of the output (default is uint8)
    """
    return (((a - min) / float(max - min)) * 255).astype(dtype)


def top_down_view_map(points,
                      side_range=(-2, 2),
                      fwd_range=(0, 4),
                      resolution=0.1):
    """ Creates an top down view representation of the point cloud data.

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
        resolution: (float) desired resolution in metres to use
                    Each output pixel will represent an square region res x res
                    in size.
    """

    points_x = points[:, 0]
    points_y = points[:, 1]
    points_z = points[:, 2]

    ff = np.logical_and((points_x > fwd_range[0]), (points_x < fwd_range[1]))
    ss = np.logical_and((points_y > side_range[0]), (points_y < side_range[1]))
    indices = np.argwhere(np.logical_and(ff, ss)).flatten()  # indices of points within the range

    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (points_y[indices] / resolution).astype(np.int32)  # x axis is -y in LIDAR
    y_img = (points_x[indices] / resolution).astype(np.int32)  # y axis is -x in LIDAR
    # will be inverted later

    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor used to prevent issues with -ve vals rounding upwards
    x_img -= int(np.floor(side_range[0] / resolution))
    y_img -= int(np.floor(fwd_range[0] / resolution))

    # FILL PIXEL VALUES IN IMAGE ARRAY
    x_max = int((side_range[1] - side_range[0]) / resolution)
    y_max = int((fwd_range[1] - fwd_range[0]) / resolution)


def birds_eye_point_cloud(points,
                          side_range=(-2, 2),
                          fwd_range=(0, 4),
                          res=0.01,
                          min_height=-2,
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
        height[height > min_height] = 1
        height[height <= min_height] = 0
        pixel_values = scale_to_255(height, min=-1, max=1)
        im[-y_img, x_img] = pixel_values
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
