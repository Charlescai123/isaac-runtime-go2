from torch import nn
import torch.nn.functional as F


class ValueMLP(nn.Module):
    def __init__(self, hidden_dim, input_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.value_func = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(self.hidden_dim, self.hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(self.hidden_dim, self.output_dim))

        nn.init.kaiming_normal_(self.value_func[0].weight)
        nn.init.kaiming_normal_(self.value_func[2].weight)
        nn.init.kaiming_normal_(self.value_func[4].weight)

    def forward(self, data):
        value = self.value_func(data)
        return value


class BellmanLoss(nn.Module):
    def __init__(self, *args):
        super().__init__()
        pass

    def forward(self, value_state, value_next, reward, discount=0.99):
        y = reward + discount * value_next
        loss = F.mse_loss(y, value_state)
        return loss
