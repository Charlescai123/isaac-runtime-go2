import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import random
import pickle
from job.offline.models.inverse_dynamics import InverseModel
from job.offline.models.buffer import ReplayBuffer, Trajectory


def set_seed(seed, deterministic_torch=False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


# TODO: add model ensemble


def train_reward(seed=100, device="cuda"):

    load_check = True   # TODO: put it in the parameter input

    # if load_check:
    #     trajectories = load(bucket=Config.bucket)
    # else:
    #     return
    with open("data/offline/normalized_dataset.dat", "rb") as f:
        data = pickle.load(f)

    num_eval = 100
    batch_size = 64

    # env_list = [gym.make(Config.dataset) for _ in range(num_eval)]

    traj_length = 2000

    path_num = len(data)
    state_dim = data[0][0].shape[1]
    action_dim = data[0][1].shape[1]
    length = [0 for i in range(num_eval)]
    traj = [Trajectory(length=traj_length, state_dim=state_dim, action_dim=action_dim) for i in range(path_num)]

    state_dim = data[0][0].shape[1]
    action_dim = data[0][1].shape[1]

    buffer_size = 20_000_000    # TODO: put it in the parameter input
    buffer = ReplayBuffer(state_dim=state_dim, action_dim=action_dim, buffer_size=buffer_size, device=device, seed=seed)

    full_data = {}
    for i in range(len(data)):
        if i == 0:
            full_data["observations"] = data[i][0]
            full_data["actions"] = data[i][1]
            if data[i][2].ndim == 1:
                full_data["rewards"] = np.expand_dims(data[i][2], axis=1)
            else:
                full_data["rewards"] = data[i][2]
            full_data["next_observations"] = data[i][3]
            temp = np.zeros(data[i][0].shape[0])
            temp[-1] = 1
            full_data["terminals"] = np.expand_dims(temp, axis=1)
        else:
            full_data["observations"] = np.concatenate((full_data["observations"], data[i][0]), axis=0)
            full_data["actions"] = np.concatenate((full_data["actions"], data[i][1]), axis=0)
            if data[i][2].ndim == 1:
                full_data["rewards"] = np.concatenate((full_data["rewards"], np.expand_dims(data[i][2], axis=1)), axis=0)
            else:
                full_data["rewards"] = np.concatenate((full_data["rewards"], data[i][2]),
                                                      axis=0)
            full_data["next_observations"] = np.concatenate((full_data["next_observations"], data[i][3]), axis=0)
            temp = np.zeros(data[i][0].shape[0])
            temp[-1] = 1
            full_data["terminals"] = np.concatenate((full_data["terminals"], np.expand_dims(temp, axis=1)), axis=0)
    buffer.load_data(full_data)

    epoch_num = 8000
    inverse_model = InverseModel(state_dim=state_dim, action_dim=action_dim).to(device=device)
    optimizer = torch.optim.Adam(inverse_model.parameters(), lr=0.0001)
    loss_fn = torch.nn.MSELoss()
    for i in range(epoch_num):
        batch = buffer.sample(batch_size=batch_size)
        batch = [b.to(device) for b in batch]
        state = batch[0]
        action_true = batch[1]
        next_state = batch[3]
        s_comb = torch.concat((state, next_state), axis=1)
        action_pred = inverse_model(s_comb)
        loss = loss_fn(action_pred, action_true)
        print(f"loss at epoch {i}", loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(inverse_model.state_dict(), "job/offline/weights/inverse_model.pt")


if __name__ == "__main__":
    train_reward()