import dataclasses
import os
import sys
import click
import random
import torch
import numpy as np
import torch.nn.functional as F
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quad_gym.env_builder import build_a1_ground_env
from quad_gym.gym_config import GymConfig
from utils.utils import load_config, save_params
from policy.policy_config import PolicyConfig
from job.job_config import JobConfig
from job.job_monitor.job_logger import Logger
from dataclasses_json import dataclass_json
from policy.policy_config import AllControlModels
from stable_baselines3.common.buffers import ReplayBuffer
import matplotlib.pyplot as plt

@dataclass_json
@dataclasses.dataclass
class AllConfig:
    GymParams: GymConfig = GymConfig()
    PolicyParams: PolicyConfig = PolicyConfig()
    JobParams: JobConfig = JobConfig()


@click.command()
@click.argument('config')
@click.option('--gpu', is_flag=True)
@click.option('--generate', is_flag=True, help="Generate a config file instead.")
@click.option('-p', '--params', nargs=2, multiple=True)
def run(**kwargs):
    if kwargs['generate']:
        save_params(AllConfig(), kwargs['config'])
        print("Config file generated.")
        return

    if not kwargs['gpu']:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    params: AllConfig = load_config(kwargs['config'], overrides=kwargs['params'])

    random.seed(params.JobParams.seed)
    np.random.seed(params.JobParams.seed)
    torch.manual_seed(params.JobParams.seed)
    torch.backends.cudnn.deterministic = params.JobParams.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and params.JobParams.cuda else "cpu")

    env = build_a1_ground_env(params.GymParams)

    control_model_name = params.PolicyParams.ControlModuleParams.control_model
    control_model_params = params.PolicyParams.ControlModuleParams.control_model_params

    agent = AllControlModels[control_model_name]["model"](control_model_params, env, device)

    logger = Logger(params.JobParams)

    logger.log_all_params(all_params=params, mode=params.JobParams.run_mode)

    if params.PolicyParams.ControlModuleParams.pretrained_model_path is not None:
        agent.load_model(params.PolicyParams.ControlModuleParams.pretrained_model_path)

    replay_buffer = ReplayBuffer(control_model_params.buffer_size,
                                 env.observation_space,
                                 env.action_space, device,
                                 handle_timeout_termination=False)

    epoch = 10

    # create a replay buffer from offline data
    with open("data/offline/unnormalized_stitched_dataset.dat", "rb") as f:
        un_normalized_data = pickle.load(f)

    for traj in un_normalized_data:
        for i in range(len(traj[0])):
            obs = traj[0][i]
            action = traj[1][i]
            reward = traj[2][i]
            next_obs = traj[3][i]
            # termination = traj[4][i]
            termination = False
            infos = []
            replay_buffer.add(obs, next_obs, action, reward, termination, infos)

    print("Replay memory size:", replay_buffer.size())
    batch_size = 128
    iters = int(replay_buffer.size() / 128)

    critic_loss_mean_list = []
    action_value_mean_list = []

    for i in range(epoch):
        epoch_critic_loss_list = []
        actor_value_list = []
        for _ in range(iters):
            minibatch = replay_buffer.sample(batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = agent.actor.get_action(minibatch.next_observations)
                qf1_next_target = agent.qf1_target(minibatch.next_observations, next_state_actions)
                qf2_next_target = agent.qf2_target(minibatch.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - agent.params.alpha * next_state_log_pi
                next_q_value = minibatch.rewards.flatten() + (1 - minibatch.dones.flatten()) * agent.params.gamma * (
                    min_qf_next_target).view(-1)

            qf1_a_values = agent.qf1(minibatch.observations, minibatch.actions).view(-1)
            qf2_a_values = agent.qf2(minibatch.observations, minibatch.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss
            agent.q_optimizer.zero_grad()
            qf_loss.backward()
            agent.q_optimizer.step()

            pi, log_pi, _ = agent.actor.get_action(minibatch.observations)
            qf1_pi = agent.qf1(minibatch.observations, pi)
            qf2_pi = agent.qf2(minibatch.observations, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
            # actor_loss = ((agent.params.alpha * log_pi) - min_qf_pi).mean()
            actor_loss = (- min_qf_pi).mean()

            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()
            actor_value_list.append(actor_loss.detach().numpy())
            epoch_critic_loss_list.append(qf_loss.detach().numpy())

        mean_critic_loss = np.mean(epoch_critic_loss_list)
        mean_action_value = np.mean(actor_value_list)
        critic_loss_mean_list.append(mean_critic_loss)
        action_value_mean_list.append(mean_action_value)

        print(f"At epoch {i + 1}: Average critic loss {np.mean(epoch_critic_loss_list)}  \
         Average action value {np.mean(actor_value_list)}")

    model_dir = logger.model_dir + 'offline'
    os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}/weights.cleanrl_model"
    torch.save((agent.actor.state_dict(), agent.qf1.state_dict(), agent.qf2.state_dict()), model_path)

    plt.plot(critic_loss_mean_list)
    plt.plot(action_value_mean_list)
    plt.title("critic_loss and action value ")
    plt.savefig("critic_loss_mean.png")


if __name__ == '__main__':
    run()
