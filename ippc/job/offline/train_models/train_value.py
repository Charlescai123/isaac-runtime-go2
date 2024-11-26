import torch
import numpy as np
import pickle
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from job.offline.models.value_model import ValueMLP, BellmanLoss
from job.offline.models.buffer import ReplayBuffer
import matplotlib.pyplot as plt
import random


def set_seed(seed, deterministic_torch=False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def load_data():
    with open("data/offline/normalized_dataset.dat", "rb") as f:
        data = pickle.load(f)
    return data


def train_val(device="cuda", seed=100):
    batch_size = 256

    data = load_data()
    state_dim = data[0][0].shape[1]
    action_dim = data[0][1].shape[1]

    set_seed(seed)
    model1 = ValueMLP(hidden_dim=256, input_dim=state_dim, output_dim=1).to(device)
    model2 = ValueMLP(hidden_dim=256, input_dim=state_dim, output_dim=1).to(device)
    # TODO: change to seed
    loss = BellmanLoss()
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=3e-4)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=3e-4)
    epoch_num = 4000

    buffer_size = 20_000_000
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

    losses = []

    for i in range(epoch_num):
        batch = buffer.sample(batch_size=batch_size)
        batch = [b.to(device) for b in batch]
        value_state_1 = model1(batch[0])
        value_next_1 = model1(batch[3])
        value_state_2 = model2(batch[0])
        value_next_2 = model2(batch[3])
        loss_1 = loss(value_state_1, value_next_1, batch[2])
        loss_2 = loss(value_state_2, value_next_2, batch[2])
        total_loss = loss_1 + loss_2

        print(f"total loss at epoch {i}", total_loss)
        losses.append(total_loss.detach().cpu().numpy())

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        total_loss.backward()
        optimizer1.step()
        optimizer2.step()

    save_models = {
        'model1': model1.state_dict(),
        'model2': model2.state_dict()
    }

    plt.plot(losses)
    # plt.yscale("log", base=10)
    plt.title("Value Function Loss Curve")
    plt.savefig("value_loss.png")

    # with open("weights/value_models.pkl", "wb") as f:
    #     pickle.dump(save_models, f)

    torch.save(save_models, "job/offline/weights/value_models.pt")


if __name__ == "__main__":
    train_val()
