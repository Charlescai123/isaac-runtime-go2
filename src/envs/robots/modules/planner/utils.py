import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def getDepth(z_n, zNear, zFar):
    z_n = 2.0 * z_n - 1.0
    z_e = 2.0 * zNear * zFar / (zFar + zNear - z_n * (zFar - zNear))
    return z_e


def scale_to_255(a, min, max, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0 - 255
        Optionally specify the data type of the output (default is uint8)
    """
    return (((a - min) / float(max - min)) * 255).astype(dtype)


def to_ply(points, colors) -> o3d.geometry.PointCloud:
    # Example: Generate a random point cloud (replace this with your actual data)
    # points = np.random.rand(100, 3)  # 100 points with x, y, z coordinates

    # Create an Open3D PointCloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    return point_cloud


def top_down_view_map(points,
                      side_range=(-2, 2),
                      fwd_range=(0, 4),
                      resolution=0.1):
    """ Creates a top-down view representation of the point cloud data.

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


def birds_eye_point_cloud(points: np.ndarray,
                          side_range=(-2, 2),
                          fwd_range=(0, 4),
                          resolution=0.01,
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
        resolution: (float) desired resolution in metres to use
                    Each output pixel will represent a square region res x res
                    in size.
        min_height:  (float)(default=-2.73)
                    Used to truncate height values to this minimum height
                    relative to the sensor (in metres).
                    The default is set to -2.73, which is 1 metre below a flat
                    road surface given the configuration in the kitti dataset.
        max_height: (float)(default=1.27)
                    Used to truncate height values to this maximum height
                    relative to the sensor (in metres).
                    The default is set to 1.27, which is 3m above a flat road
                    surface given the configuration in the kitti dataset.
        as_occupancy: (boolean)(default=False)
                      To generate an occupancy map or not
    """
    x_lidar = points[:, 0]
    y_lidar = points[:, 1]
    z_lidar = points[:, 2]

    max_x = max(x_lidar)
    min_x = min(x_lidar)
    max_y = max(y_lidar)
    min_y = min(y_lidar)
    max_z = max(z_lidar)
    min_z = min(z_lidar)
    print(f"max_x: {max_x}")
    print(f"min_x: {min_x}")
    print(f"max_y: {max_y}")
    print(f"min_y: {min_y}")
    print(f"max_z: {max_z}")
    print(f"min_z: {min_z}")
    fwd_range = [min_x, max_x]
    side_range = [min_y, max_y]
    min_height = min(z_lidar)
    max_height = max(z_lidar)

    ff = np.logical_and((x_lidar > fwd_range[0]), (x_lidar < fwd_range[1]))
    ss = np.logical_and((y_lidar > side_range[0]), (y_lidar < side_range[1]))
    indices = np.argwhere(np.logical_and(ff, ss)).flatten()

    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (-y_lidar[indices] / resolution).astype(np.int32)  # x axis is -y in LIDAR
    y_img = (x_lidar[indices] / resolution).astype(np.int32)  # y axis is -x in LIDAR
    # will be inverted later

    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor used to prevent issues with -ve vals rounding upwards
    x_img -= int(np.floor(side_range[0] / resolution))
    y_img -= int(np.floor(fwd_range[0] / resolution))

    # FILL PIXEL VALUES IN IMAGE ARRAY
    x_max = int((side_range[1] - side_range[0]) / resolution)
    y_max = int((fwd_range[1] - fwd_range[0]) / resolution)
    print(f"x_img: {x_img}")
    print(f"y_img: {y_img}")
    print(f"len of x_img: {len(x_img)}")
    print(f"len of y_img: {len(y_img)}")
    print(f"x_max: {x_max}")
    print(f"y_max: {y_max}")
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

        # RESCALE THE HEIGHT VALUES - to be between the range 0 - 255
        pixel_values = scale_to_255(pixel_values, min=min_height, max=max_height)
        im = np.zeros([y_max, x_max], dtype=np.uint8)
        im[-y_img, x_img] = pixel_values  # -y because images start from top left
    print(f"im: {im}")
    print(f"im: {im.shape}")

    return im


def path_plot(distance_map, path, start, goal):
    """Plot the path from start point to goal point on a distance map"""
    # Visualize the distance map
    plt.figure(figsize=(8, 8))
    # plt.imshow(self._costmap_for_plot, cmap="coolwarm", origin="lower")
    plt.imshow(distance_map, cmap='viridis')
    plt.gca().invert_yaxis()  # Inverse y-axis
    plt.gca().invert_xaxis()  # Inverse x-axis
    plt.gca().yaxis.tick_right()  # move y-axis to the right
    plt.colorbar(label="Distance from Goal (m)")
    plt.scatter(*goal[::-1], color="green", label="Goal", marker="x", s=100)
    plt.scatter(*start[::-1], color="blue", label="Start", marker="o", s=100)

    # Visualize the shortest path
    plt.plot(np.asarray(path)[:, 1], np.asarray(path)[:, 0], color="black", linewidth=2,
             label="Shortest Path")
    plt.legend()
    plt.title("Shortest Path on Distance Map")
    plt.show()


def get_shortest_path(distance_map, start_pt, goal_pt, precise=0.01, max_cnt=1000):
    """Given the start and goal point on a distance map, find the shortest path through back-tracing

    Parameters
    ----------
    distance_map : 2d array/list
                   a gridmap with corresponding distances at each grid to the goal point. The obstacles
                   on the map should be assigned a large distance value

    start_pt: The coordinate of a start point on the map, must be integer with order in (y, x)
    goal_pt: The coordinate of a goal point on the map, must be integer with order in (y, x)
    precise: The precise for the searching algorithm (gradient)
    max_cnt: A number used to search for the largest times to prevent local minima on map

    Returns
    ----------
    shortest_path: a list of all the points from start to the goal, with each point order in (y, x)
    """
    print(f"Searching for the shortest path...")

    # Assure the point of Int type
    start_pt = (int(start_pt[0]), int(start_pt[1]))
    goal_pt = (int(goal_pt[0]), int(goal_pt[1]))

    # Gradient of the map (∇distance_map)
    gy, gx = np.gradient(distance_map)
    gy += np.random.uniform(-precise, precise, size=gy.shape)
    gx += np.random.uniform(-precise, precise, size=gx.shape)
    # np.savetxt("gx.txt", gx)
    # np.savetxt("gy.txt", gy)

    shortest_path = [start_pt]
    current = start_pt
    cnt = 0
    while np.linalg.norm(np.array(current) - np.array(goal_pt)) > 1:
        # Local minima
        if cnt >= max_cnt:
            break
        # print(f"current: {current}")
        y, x = current
        dy, dx = gy[y, x], gx[y, x]  # map gradient

        # Avoid local-minima
        if dy == 0 and dx == 0:
            break

        # Normalized direction
        step = np.array([-dy, -dx]) / np.linalg.norm([dy, dx]) + 1e-6
        step = [step[0].item(), step[1].item()]

        # Move to the next point and continue tracing
        next_point = [int(round(y + step[0])), int(round(x + step[1]))]

        # Exit when tracing to the start point
        if next_point == current:
            break

        # Otherwise continue tracing
        current = next_point

        # Append next point to the result
        shortest_path.append(next_point)

        cnt += 1

    return shortest_path
