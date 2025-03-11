import torch
import numpy as np


@torch.jit.script
def quat_to_rot_mat(q):
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    Nq = w * w + x * x + y * y + z * z
    s = 2.0 / Nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ = y * Y, y * Z
    zZ = z * Z

    rotation_matrix = torch.stack([
        torch.stack([1.0 - (yY + zZ), xY - wZ, xZ + wY], dim=-1),
        torch.stack([xY + wZ, 1.0 - (xX + zZ), yZ - wX], dim=-1),
        torch.stack([xZ - wY, yZ + wX, 1.0 - (xX + yY)], dim=-1)
    ], dim=-2)

    return rotation_matrix


@torch.jit.script
def copysign(a, b):
    # type: (float, Tensor) -> Tensor
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)


@torch.jit.script
def get_euler_zyx_from_quaternion(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
                q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(
        torch.abs(sinp) >= 1, copysign(np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
                q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack(
        (roll % (2 * np.pi), pitch % (2 * np.pi), yaw % (2 * np.pi)), dim=1)


if __name__ == '__main__':
    q = torch.tensor([[0.0, 0.0, 0.707, 0.707]])
    import math
    import pybullet as p

    roll = math.radians(30)
    pitch = math.radians(45)
    yaw = math.radians(60)

    quaternion = p.getQuaternionFromEuler([roll, pitch, yaw])
    quaternion = torch.tensor([quaternion])
    print(f"quaternion: {quaternion}")
    import torch


    def quaternion_to_euler_xyz(q):
        """
        """
        roll = torch.atan2(2 * (q[:, 3] * q[:, 0] + q[:, 1] * q[:, 2]),
                           1 - 2 * (q[:, 0] ** 2 + q[:, 1] ** 2))
        pitch = torch.asin(torch.clamp(2 * (q[:, 3] * q[:, 1] - q[:, 2] * q[:, 0]), -1.0, 1.0))
        yaw = torch.atan2(2 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1]),
                          1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2))
        return torch.stack((roll, pitch, yaw), dim=-1)


    def quaternion_to_euler_zyx(q):
        """
        """
        yaw = torch.atan2(2 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1]),
                          1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2))
        pitch = torch.asin(torch.clamp(-2 * (q[:, 0] * q[:, 2] - q[:, 3] * q[:, 1]), -1.0, 1.0))
        roll = torch.atan2(2 * (q[:, 3] * q[:, 0] + q[:, 1] * q[:, 2]),
                           1 - 2 * (q[:, 0] ** 2 + q[:, 2] ** 2))
        return torch.stack((roll, pitch, yaw), dim=-1)


    print(f"zyx: {quaternion_to_euler_zyx(quaternion)}")
    print(f"xyz: {quaternion_to_euler_xyz(quaternion)}")

    quaternion2 = torch.tensor([[0.7071, 0.7071, 0, 0]])
    euler_angles = get_euler_zyx_from_quaternion(quaternion)
    print("Euler Angles (roll, pitch, yaw):", euler_angles)
