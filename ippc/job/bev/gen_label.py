import cv2
import math
# import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
from nuscenes.utils.data_classes import Box
import torch
from nuscenes.nuscenes import NuScenes
import os

xbound = [-2.0, 2.0, 0.005]  # dm
ybound = [-2.0, 2.0, 0.005]
zbound = [-1.0, 1.0, 2.0]
dbound = [4.0, 45.0, 1.0]

grid_conf = {
    'xbound': xbound,
    'ybound': ybound,
    'zbound': zbound,
    'dbound': dbound,
}

# nusc = NuScenes(version='v1.0-{}'.format(version),
#                 dataroot=os.path.join(dataroot, version),
#                 verbose=False)

version = 'mini'
dataroot = '/home/hongi/devel/lift-splat-shoot/data/nuscenes'


# nusc = NuScenes(version='v1.0-{}'.format(version),
#                 dataroot=os.path.join(dataroot, version),
#                 verbose=False)


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])

    return dx, bx, nx


# def get_binimg(rec):
#     egopose = nusc.get('ego_pose', nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
#
#     print(egopose)
#
#     trans = -np.array(egopose['translation'])
#     rot = Quaternion(egopose['rotation']).inverse
#
#     dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
#     dx, bx, nx = dx.numpy(), bx.numpy(), nx.numpy()
#
#     img = np.zeros((nx[0], nx[1]))
#     for tok in rec['anns']:
#         inst = nusc.get('sample_annotation', tok)
#         if not inst['category_name'].split('.')[0] == 'vehicle':
#             continue
#         box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
#         box.translate(trans)
#         box.rotate(rot)
#
#         pts = box.bottom_corners()[:2].T
#         pts = np.round(
#             (pts - bx[:2] + dx[:2] / 2.) / dx[:2]
#         ).astype(np.int32)
#         pts[:, [1, 0]] = pts[:, [0, 1]]
#         cv2.fillPoly(img, [pts], 1.0)
#
#     return torch.Tensor(img).unsqueeze(0)


if __name__ == '__main__':
    num_images = len(os.listdir("./data/perception/bev/label/"))
    scene_object_label = np.load('./data/perception/bev/scene_object_label.npy', allow_pickle=True)
    print(scene_object_label)

    for id_img in range(num_images):
        label = np.load(f'./data/perception/bev/label/label_{id_img}.npy', allow_pickle=True)
        segmentation = np.load(f'./data/perception/bev/seg/seg_{id_img}.npy', allow_pickle=True)
        print(np.min(segmentation))
        cam_pos = label.item()['cam_pos']
        cam_rot = label.item()['cam_orn']
        R_cam_matrix = R.from_quat(cam_rot).as_matrix()

        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        dx, bx, nx = dx.numpy(), bx.numpy(), nx.numpy()
        img = np.zeros((nx[0], nx[1]))

        for obj in scene_object_label:
            if len(np.where(segmentation == obj['id'])[0]) == 0:
                print(f"obj_{obj['id']}_is not visible")
                continue
            print("=======================================")
            print("obj_position", obj['position'])
            print("obj_rotation", obj['rotation'])
            print("cam_pose", cam_pos)
            obj_rotation = obj['rotation']
            angle = R.from_quat(obj_rotation).as_euler('xyz', degrees=True)[-1]
            print(R.from_quat(obj_rotation).as_euler('xyz', degrees=True)[-1])
            bbox_rotation = R.from_euler('x', angle).as_quat()
            obj_position = obj['position']
            R_obj_matrix = R.from_quat(obj_rotation).as_matrix()
            trans = np.linalg.inv(R_obj_matrix) @ (np.subtract(obj_position, cam_pos))
            rotation = np.linalg.inv(R_obj_matrix) @ R_cam_matrix
            # rotation = R.from_matrix(rotation).as_quat()
            size = np.array(obj['size'])
            # w, l, h = size
            # half_w = 0.5 * w
            # half_l = 0.5 * l
            # half_h = 0.5 * h
            # corner_of_box = np.array([[-1 * half_w, -1 * half_l, -1 * half_h],
            #                           [1 * half_w, -1 * half_l, -1 * half_h],
            #                           [-1 * half_w, 1 * half_l, -1 * half_h],
            #                           [-1 * half_w, -1 * half_l, 1 * half_h],
            #                           [1 * half_w, 1 * half_l, -1 * half_h],
            #                           [1 * half_w, -1 * half_l, 1 * half_h],
            #                           [1 * half_w, 1 * half_l, -1 * half_h],
            #                           [1 * half_w, 1 * half_l, 1 * half_h],
            #                           ])
            # corner_of_box_tran = corner_of_box @ rotation
            # corner_of_box = corner_of_box + trans
            # corner_of_box_x_y = corner_of_box[:, :2]
            #
            # print(corner_of_box_x_y)
            # print(Quaternion(obj_rotation))

            box = Box(obj_position, size, Quaternion(bbox_rotation))
            print(box)
            box.translate(-1 * np.array(cam_pos))
            box.rotate(Quaternion(cam_rot).inverse)
            pts = box.bottom_corners()[:2].T
            print(pts)
            pts = np.round(
                (pts - bx[:2] + dx[:2] / 2.) / dx[:2]
            ).astype(np.int32)
            print(pts)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            print(pts)
            cv2.fillPoly(img, [pts], 1.0)

        robot_size = [0.5, 0.5, 0.5]
        q = R.from_euler('x', 45, degrees=True).as_quat()
        robot_box = Box([0, 0, 0], robot_size, Quaternion(q))
        # robot_box.rotate(Quaternion(q))
        pts = robot_box.bottom_corners()[:2].T
        pts = np.round(
            (pts - bx[:2] + dx[:2] / 2.) / dx[:2]
        ).astype(np.int32)
        pts[:, [1, 0]] = pts[:, [0, 1]]
        cv2.fillPoly(img, [pts], 0.5)

        bin_img_vis = np.array(img).squeeze()
        plt.imshow(bin_img_vis)
        plt.savefig(f'data/perception/bev/gt_result/gt_{id_img}.png'.format())
