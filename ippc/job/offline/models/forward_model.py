from torch import nn
import numpy as np
import math
import torch


class ForwardDynamics(nn.Module):
    def __init__(self, state_dim, hidden_dim=200):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(nn.Linear(self.state_dim, self.hidden_dim), nn.ReLU(),
                                 nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
                                 nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
                                 nn.Linear(self.hidden_dim, 2 * self.state_dim))
        # nn.init.normal_(self.net[0].weight, mean=0.0, std=1.0 / math.sqrt(5))
        # nn.init.normal_(self.net[2].weight, mean=0.0, std=1.0 / math.sqrt(5))
        # nn.init.normal_(self.net[4].weight, mean=0.0, std=1.0 / math.sqrt(5))
        # nn.init.normal_(self.net[6].weight, mean=0.0, std=1.0 / math.sqrt(5))
        nn.init.kaiming_normal_(self.net[0].weight)
        nn.init.kaiming_normal_(self.net[2].weight)
        nn.init.kaiming_normal_(self.net[4].weight)
        nn.init.kaiming_normal_(self.net[6].weight)

    def forward(self, state):
        out = self.net(state)
        return out
    

class ForwardLoss(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, comb, state_dim, next_state, device):
        mean = comb[:, :state_dim]
        std = comb[:, state_dim:]
        std_square = torch.square(std)
        det = torch.sum(torch.log(std_square), dim=1)
        var = torch.zeros((std.size(0), std.size(1), std.size(1))).to(device)
        var.as_strided(std.size(), [var.stride(0), var.size(2) + 1]).copy_(std)
        var = torch.square(var)
        var_inv = torch.linalg.inv(var)
        # temp = torch.bmm((mean - next_state), var_inv)
        temp = torch.einsum("bm,bmn->bn", [(mean - next_state), var_inv])
        # temp1 = torch.bmm(temp, (mean - next_state)) + torch.log(torch.linalg.det(var))
        temp1 = torch.einsum("bn,bn->b", [temp, (mean - next_state)]) + det
        # print(temp1)
        # print(var)
        loss = torch.mean(temp1, dim=0)
        return loss

