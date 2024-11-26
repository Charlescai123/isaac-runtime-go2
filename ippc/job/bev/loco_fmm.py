import copy

import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy.ndimage.morphology import binary_dilation
from skimage.measure import block_reduce
from scipy.interpolate import NearestNDInterpolator
from scipy.ndimage import median_filter
from scipy.interpolate import griddata
from policy.planner.FMM import FMMPlanner, Spline


def scale_to_255(a, min, max, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-255
        Optionally specify the data type of the output (default is uint8)
    """
    return (((a - min) / float(max - min)) * 255).astype(dtype)


f, axarr = plt.subplots(2, 2, figsize=(16, 16))
plt.axis('off')
plt.tight_layout(pad=0)

pcld = np.load("pcld.npy")

## genearate a grid world

fwd_range = (0, 4)  ## x value boundary in world measure
side_range = (-2, 2)  ## y value boundary in world
resolution = 0.01  ## each grid will represent 0.01 meter
min_height = -1
max_height = 1

image_H = int((fwd_range[1] - fwd_range[0]) / resolution)  # This gives us the grid world
image_W = int((side_range[1] - side_range[0]) / resolution)  # The origin (0, 0) of the image is at the left up corner

# create a grid world
im_grid = np.zeros([image_H, image_W], dtype=np.uint8)

points_x = pcld[:, 0]
points_y = pcld[:, 1]
points_z = pcld[:, 2]

# filter out the points that are out of the boundary

ff = np.logical_and((points_x > fwd_range[0]), (points_x < fwd_range[1]))
ss = np.logical_and((points_y > -side_range[1]), (points_y < -side_range[0]))
indices = np.argwhere(
    np.logical_and(ff, ss)).flatten()  # this will return the indices for the valid points (within the boundary)

# CONVERT TO PIXEL POSITION VALUES - Based on resolution
x_img = (-points_y[indices] / resolution).astype(np.int32)  # x axis is -y in LIDAR
y_img = (points_x[indices] / resolution).astype(np.int32)  # y axis is -x in LIDAR
# will be inverted later

# SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
# floor used to prevent issues with -ve vals rounding upwards
x_img -= int(np.floor(side_range[0] / resolution))
y_img -= int(np.floor(fwd_range[0] / resolution))
z_height = points_z[indices]

pcld_project_2d = im_grid.copy()
z_height_rescale = scale_to_255(z_height.copy(), min=min_height, max=max_height)
pcld_project_2d[-y_img, x_img] = z_height_rescale

occupancy_grid = np.ones([image_H, image_W], dtype=np.uint8) * np.nan
occupancy_grid[-y_img, x_img] = z_height

x, y = np.indices(occupancy_grid.shape)

interp_grid = np.array(occupancy_grid)
interp_grid[np.isnan(interp_grid)] = griddata((x[~np.isnan(interp_grid)], y[~np.isnan(interp_grid)]),  # points we know
                                              interp_grid[~np.isnan(interp_grid)],
                                              (x[np.isnan(interp_grid)], y[np.isnan(interp_grid)]))
z_height_feasible_mask = np.logical_and((interp_grid.copy() > -0.1), (interp_grid.copy() < 0.1))
occupancy_map = np.ones([image_H, image_W], dtype=np.uint8) * z_height_feasible_mask
# z_height_feasible = z_height_feasible_mask.astype(np.int8)
# occupancy_grid[-y_img, x_img] = z_height_feasible
# occupancy_map = median_filter(occupancy_grid, size=(7, 7))
occupancy_map = 1 - occupancy_map

# generate a cost map using FMM


planner = FMMPlanner(occupancy_map)
target_pose = [100, 100]
dd = planner.set_goal(target_pose)
distance_map = scale_to_255(np.array(dd), min=0, max=800)

curr_pos = [25, 25]
yaw = 0
curr_vel = [0, 0]

p1 = axarr[0, 0].imshow(pcld_project_2d)
p2 = axarr[0, 1].imshow(occupancy_grid)
p3 = axarr[1, 0].imshow(occupancy_map)
p4 = axarr[1, 1].imshow(distance_map)
plt.pause(100)
