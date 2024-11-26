from torch import nn
import numpy as np
import math
import torch.nn.functional as F


class RewardModel(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(nn.Linear(2 * self.state_dim, self.hidden_dim), 
                                 nn.ReLU(), 
                                 nn.Linear(self.hidden_dim, self.hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(self.hidden_dim, 1))

        nn.init.kaiming_normal_(self.net[0].weight)
        nn.init.kaiming_normal_(self.net[2].weight)
        nn.init.kaiming_normal_(self.net[4].weight)

    def forward(self, s_comb):
        reward = self.net(s_comb)
        return reward