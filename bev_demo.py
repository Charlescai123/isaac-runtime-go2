import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate a sample point cloud
# Replace this with your actual point cloud
# Point cloud format: [x, y, z]
# num_points = 10000
# point_cloud = np.random.uniform(-10, 10, (num_points, 3))  # Random points in [-10, 10] for x, y, z
pcd = o3d.io.read_point_cloud("pcd.ply")
point_cloud = np.asarray(pcd.points)

# Step 2: Define Region of Interest (ROI)
x_min, x_max = 45, 70
y_min, y_max = -10, 10
z_min, z_max = -20, 20  # Optional filtering by height

# Filter points within the ROI
roi_points = point_cloud[
    (point_cloud[:, 0] >= x_min) & (point_cloud[:, 0] <= x_max) &
    (point_cloud[:, 1] >= y_min) & (point_cloud[:, 1] <= y_max) &
    (point_cloud[:, 2] >= z_min) & (point_cloud[:, 2] <= z_max)
]

# Step 3: Define grid parameters for BEV map
grid_size = 0.1  # Size of each grid cell
x_bins = int((x_max - x_min) / grid_size)
y_bins = int((y_max - y_min) / grid_size)

# Initialize BEV map (height map)
bev_map = np.zeros((x_bins, y_bins))

# Step 4: Populate BEV map with max height
for point in roi_points:
    x, y, z = point
    x_idx = int((x - x_min) / grid_size)
    y_idx = int((y - y_min) / grid_size)
    bev_map[x_idx, y_idx] = max(bev_map[x_idx, y_idx], z)  # Use max height for the grid cell

# Step 5: Visualize BEV Map
plt.figure(figsize=(10, 8))
plt.imshow(bev_map.T, cmap='viridis', origin='lower', extent=[x_min, x_max, y_min, y_max])
plt.colorbar(label='Height (m)')
plt.title('BEV Map (Height Map)')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.show()
