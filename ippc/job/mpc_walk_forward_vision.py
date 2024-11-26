import os
import sys
import click

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quad_gym.env_builder import build_a1_ground_env
from quad_gym.gym_config import GymConfig
from utils.utils import load_config, save_params
from policy.control.mpc.MPC_Control import MPC, MPCConfig
from policy.perception.utils import *


@click.command()
@click.argument('config')
@click.option('--gpu', is_flag=True)
@click.option('--generate', is_flag=True, help="Generate a config file instead.")
@click.option('-p', '--params', nargs=2, multiple=True)
def run(**kwargs):
    if kwargs['generate']:
        save_params(GymConfig(), kwargs['config'])
        print("Config file generated.")
        return

    if not kwargs['gpu']:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    params: GymConfig = load_config(kwargs['config'], overrides=kwargs['params'])
    env = build_a1_ground_env(params)
    mpc = MPC(MPCConfig(), env.robot)

    rgb_save_folder = './data/perception/bev/rgb'
    depth_save_folder = './data/perception/bev/depth'
    seg_save_folder = './data/perception/bev/seg'
    label_save_folder = './data/perception/bev/bev'

    os.makedirs(rgb_save_folder, exist_ok=True)
    os.makedirs(depth_save_folder, exist_ok=True)
    os.makedirs(seg_save_folder, exist_ok=True)
    os.makedirs(label_save_folder, exist_ok=True)

    show_plot = True
    save_data = True
    episodes = 1

    stepX = 1
    stepY = 1
    near_val = 0.1
    far_val = 5

    if show_plot:
        f, axarr = plt.subplots(2, 2, figsize=(16, 16))
        plt.axis('off')
        plt.tight_layout(pad=0)

    for i in range(episodes):
        current_time = env.robot.GetTimeSinceReset()
        env.reset()
        mpc.reset(env.robot)
        for j in range(1000):
            lin_speed, ang_speed = mpc.generate_example_linear_angular_speed(current_time)
            lin_speed = (0.5, 0., 0.)
            mpc.update_controller_params(lin_speed, ang_speed)
            mpc.controller.update()
            action, info = mpc.controller.get_action()

            if params.RobotParams.motor_control_mode == 'torque':
                action = mpc.convert2torque(action, env.robot)
            _, _, ter, trunc, _ = env.step(action)

            rgb, dep, seg, info = env.get_vision_observation(return_label=True)
            projection_matrix = info["projection_matrix"]
            view_matrix = info["view_matrix"]
            imgW = info["width"]
            imgH = info["height"]
            camPos = info["cam_pos"]

            current_time = env.robot.GetTimeSinceReset()
            realDepthImg = dep.copy()

            for w in range(0, imgW, stepX):
                for h in range(0, imgH, stepY):
                    realDepthImg[w][h] = getDepth(dep[w][h], near_val, far_val)

            pointCloud = np.empty([np.int32(imgH / stepY), np.int32(imgW / stepX), 4])

            projectionMatrix = np.asarray(projection_matrix).reshape([4, 4], order='F')

            viewMatrix = np.asarray(view_matrix).reshape([4, 4], order='F')

            tran_pix_world = np.linalg.inv(np.matmul(projectionMatrix, viewMatrix))

            for h in range(0, imgH, stepY):
                for w in range(0, imgW, stepX):
                    x = (2 * w - imgW) / imgW
                    y = -(2 * h - imgH) / imgH  # be careful！ deepth and its corresponding position
                    z = 2 * dep[h, w] - 1
                    # z = realDepthImg[h,w]
                    pixPos = np.asarray([x, y, z, 1])
                    # print(pixPos)
                    position = np.matmul(tran_pix_world, pixPos)
                    pointCloud[np.int32(h / stepY), np.int32(w / stepX), :] = position / position[3]

            # Here the point cloud is in the world frame
            pointCloud = np.reshape(pointCloud[:, :, :3], newshape=(-1, 3))
            # Transform the point cloud to the robot frame
            # (we assume that the camera lies on the origin of the robot frame)
            pointCloud -= np.array([camPos])

            # we further transform the point cloud to the frame at the robot_feet by adding the height h_t
            pointCloud[:, 2] += camPos[2]

            # we then project the point cloud onto the grid world

            np.save("pcld", pointCloud)

            bev_img = birds_eye_point_cloud(pointCloud, min_height=-2, max_height=2)

            if show_plot:
                axarr[0, 0].imshow(rgb)
                axarr[0, 1].imshow(realDepthImg)
                axarr[1, 0].imshow(seg)
                axarr[1, 1].imshow(bev_img)
                plt.pause(0.1)

            if save_data:
                rgb_save_path = os.path.join(rgb_save_folder, f"rgb_{j}.png")
                depth_save_path = os.path.join(depth_save_folder, f"depth_{j}.png")
                seg_save_path = os.path.join(seg_save_folder, f"seg_{j}.png")
                bev_save_path = os.path.join(label_save_folder, f"bev_{j}.png")

                plt.imsave(rgb_save_path, rgb)
                plt.imsave(depth_save_path, realDepthImg)
                plt.imsave(seg_save_path, seg)
                plt.imsave(bev_save_path, bev_img)

if __name__ == '__main__':
    run()
