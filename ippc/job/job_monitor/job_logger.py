import os
import numpy as np
import shutil
import distutils.util
import matplotlib.pyplot as plt
import io
import PIL.Image
import cv2
from job.job_config import JobConfig
from torch.utils.tensorboard import SummaryWriter


class EvalResultType:
    images = 'images'  # list of images
    scalar = 'scalar'  # one scalar


class EvalResult:
    name: str
    type: EvalResultType
    data: None

    def __init__(self, name, type, data):
        self.name = name
        self.type = type
        self.data = data


class Logger:
    def __init__(self, params: JobConfig):
        self.params = params
        self.log_dir = 'runs/' + self.params.job_name
        self.model_dir = 'runs/' + self.params.job_name + '/models/'
        self.clear_cache()
        self._training_log_writer = SummaryWriter(self.log_dir + '/training')
        self._evaluation_log_writer = SummaryWriter(self.log_dir + '/eval')

        self.episode_reward_list = []
        self.episode_position_list = []
        self.episode_velocity_list = []
        self.episode_orientation_list = []
        self.episode_critic_loss_list = []
        self.episode_actor_loss_list = []
        self.episode_q_values_list = []
        self.episode_accumulated_reward = 0.
        self.episode_mean_critic_loss = 0
        self.global_steps = 0
        self.episode_run_steps = 0

    def log_step_performance_info(self, reward, robot_states_dict):
        self.episode_reward_list.append(reward)
        self.episode_position_list.append(robot_states_dict["position"])
        self.episode_velocity_list.append(robot_states_dict["orientation"])
        self.episode_orientation_list.append(robot_states_dict["velocity"])

    def log_step_loss_info(self, loss_dict):
        self.episode_critic_loss_list.append(loss_dict["critic_loss"])
        self.episode_actor_loss_list.append(loss_dict["actor_loss"])
        self.episode_q_values_list.append(loss_dict["q_values"])

    def get_episode_info(self, termination, mode='train'):
        acc_reward = np.sum(self.episode_reward_list)
        run_steps = len(self.episode_velocity_list)

        mean_x_y_z = np.mean(self.episode_position_list, axis=0)
        std_x_y_z = np.std(self.episode_position_list, axis=0)
        mean_r_p_y = np.mean(self.episode_orientation_list, axis=0)
        std_r_p_y = np.std(self.episode_orientation_list, axis=0)

        survive = (1 - termination)

        res = [
            EvalResult('/acc_reward', EvalResultType.scalar, acc_reward),
            EvalResult('/run_steps', EvalResultType.scalar, run_steps),
            EvalResult('/X_mean_world', EvalResultType.scalar, mean_x_y_z[0]),
            EvalResult('/X_std_world', EvalResultType.scalar, std_x_y_z[0]),
            EvalResult('/Y_mean_world', EvalResultType.scalar, mean_x_y_z[1]),
            EvalResult('/Y_std_world', EvalResultType.scalar, std_x_y_z[1]),
            EvalResult('/Z_mean_world', EvalResultType.scalar, mean_x_y_z[2]),
            EvalResult('/Z_std_world', EvalResultType.scalar, std_x_y_z[2]),
            EvalResult('/r_mean_body', EvalResultType.scalar, mean_r_p_y[0]),
            EvalResult('/r_std_body', EvalResultType.scalar, std_r_p_y[0]),
            EvalResult('/p_mean_body', EvalResultType.scalar, mean_r_p_y[1]),
            EvalResult('/p_std_body', EvalResultType.scalar, std_r_p_y[1]),
            EvalResult('/y_mean_body', EvalResultType.scalar, mean_r_p_y[2]),
            EvalResult('/y_std_body', EvalResultType.scalar, std_r_p_y[2]),
            EvalResult('/survive', EvalResultType.scalar, survive),
        ]

        self.episode_accumulated_reward = acc_reward

        self.episode_run_steps = run_steps

        if mode == 'train':
            mean_critic_loss = np.mean(self.episode_critic_loss_list, axis=0)

            res.append(EvalResult('/mean_critic_loss', EvalResultType.scalar, mean_critic_loss))

            mean_actor_loss = np.mean(self.episode_actor_loss_list, axis=0)
            res.append(EvalResult('/mean_actor_loss', EvalResultType.scalar, mean_actor_loss))

            mean_q_value = np.mean(self.episode_q_values_list, axis=0)
            res.append(EvalResult('/mean_q_value', EvalResultType.scalar, mean_q_value))

            self.episode_mean_critic_loss = mean_critic_loss
        else:
            trajectory_image = self.plot_xyz_trajectory()

            res.append(EvalResult('/trajectory', EvalResultType.images, trajectory_image))

        return res

    def reset_log_info(self):
        self.episode_reward_list = []
        self.episode_position_list = []
        self.episode_velocity_list = []
        self.episode_orientation_list = []
        self.episode_critic_loss_list = [0.]
        self.episode_actor_loss_list = [0.]
        self.episode_q_values_list = [0.]
        self.episode_accumulated_reward = 0.
        self.episode_mean_critic_loss = 0
        self.episode_run_steps = 0

    def log_run_results(self, ep, termination, mode='train'):

        run_results = self.get_episode_info(termination, mode)

        name = self.params.job_name

        if mode == 'train':
            log_writer = self._training_log_writer
            print(f"Training at {ep} episodes: accumulated reward: {self.episode_accumulated_reward:.6},"
                  f"critic_loss: {self.episode_mean_critic_loss:.6}, steps_ep: {self.episode_run_steps}")

        elif mode == 'eval':
            log_writer = self._evaluation_log_writer
            print(f"Evaluation at {ep} episodes: accumulated reward: {self.episode_accumulated_reward:.6},"
                  f" steps_ep: {self.episode_run_steps} ")
        else:
            print(f"Unknown mode")
            raise NameError

        for result in run_results:
            if result.type == EvalResultType.images:
                log_writer.add_image(f"{result.name}/",
                                     np.array(result.data),
                                     global_step=self.global_steps,
                                     dataformats="HWC")
            elif result.type == EvalResultType.scalar:
                log_writer.add_scalar(f"{result.name}/", result.data, global_step=self.global_steps)
            else:
                print(f"Unknown performance result: {result}")

    def log_all_params(self, all_params, mode):
        if mode == 'train':
            log_writer = self._training_log_writer
        elif mode == 'test':
            log_writer = self._evaluation_log_writer
        else:
            print(f"Unknown mode")
            raise NameError
        log_writer.add_text(
        "Allparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(all_params).items()])),
        )

    def plot_xyz_trajectory3d(self):
        ax = plt.figure().add_subplot(projection='3d')
        x = np.array(self.episode_position_list[:, 0])
        y = np.array(self.episode_position_list[:, 1])
        z = np.array(self.episode_position_list[:, 2])
        ax.plot(x, y, z, label='Moving trajectory (XYZ)')
        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=20., azim=-45)
        title = 'Acc_reward: {} Steps:{}'.format(int(self.episode_accumulated_reward), self.episode_run_steps)
        ax.set_title(title)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # plt.close(ax)
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = np.array(image)

        return image

    def plot_xyz_trajectory(self):
        figure = plt.figure(figsize=(9, 4))

        plt.subplot(1, 2, 1)
        plt.xlabel('X position')
        plt.ylabel('Y position')

        x = np.array(self.episode_position_list)[:, 0]
        y = np.array(self.episode_position_list)[:, 1]
        z = np.array(self.episode_position_list)[:, 2]

        plt.scatter(x[0], y[0], label='Start', marker="*", c='g', s=100, zorder=2)
        plt.scatter(x[-1], y[-1], label='End', marker="*", c='b', s=100, zorder=2)

        plt.plot(x, y, label='Trajectory in XY Plane')
        plt.legend(loc='best', fontsize='small')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(np.arange(self.episode_run_steps), z, label='Height change')
        plt.xlabel('Steps')
        plt.ylabel('Height')
        plt.legend(loc='best', fontsize='small')
        plt.grid(True)

        plt.legend(loc='best', fontsize='x-small')
        title = 'Acc Reward: {} Steps: {}'.format(int(self.episode_accumulated_reward), self.episode_run_steps)
        plt.suptitle(title)
        plt.grid(True)

        figure.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(figure)
        buf.seek(0)
        # image = tf.image.decode_png(buf.getvalue(), channels=4)
        # image = tf.expand_dims(image, 0)
        image = PIL.Image.open(buf)
        image = np.array(image)

        return image

    def clear_cache(self):
        if os.path.isdir(self.log_dir):
            if self.params.force_override:
                shutil.rmtree(self.log_dir)
            else:
                print(self.log_dir, 'already exists.')
                resp = input('Override log file? [Y/n]\n')
                if resp == '' or distutils.util.strtobool(resp):
                    print('Deleting old log dir')
                    shutil.rmtree(self.log_dir)
                else:
                    print('Okay bye')
                    exit(1)


def plot_trajectory(trajectory_tensor, reference_trajectory_tensor=None):
    """
   trajectory_tensor: an numpy array [n, 4], where n is the length of the trajectory,
                       5 is the dimension of each point on the trajectory, containing [x, x_dot, theta, theta_dot]
   """
    trajectory_tensor = np.array(trajectory_tensor)
    reference_trajectory_tensor = np.array(reference_trajectory_tensor)
    n, c = trajectory_tensor.shape

    y_label_list = ["x", "x_dot", "theta", "theta_dot"]

    plt.figure(figsize=(9, 6))

    for i in range(c):

        plt.subplot(c, 1, i + 1)
        plt.plot(np.arange(n), trajectory_tensor[:, i], label=y_label_list[i])

        if reference_trajectory_tensor is not None:
            plt.plot(np.arange(n), reference_trajectory_tensor[:, i], label=y_label_list[i])

        plt.legend(loc='best')
        plt.grid(True)

    plt.tight_layout()
    plt.savefig("trajectory.png", dpi=300)



