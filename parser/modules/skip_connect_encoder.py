import torch
import torch.nn as nn


class SkipConnectEncoder(nn.Module):
    def __init__(self, hidden_size):
        super(SkipConnectEncoder, self).__init__()
        self.hidden_size = hidden_size

        # To encode valence information
        self.HASCHILD_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.NOCHILD_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.valence_linear = nn.Linear(self.hidden_size, self.hidden_size)

        # To encode direction information
        self.LEFT_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.RIGHT_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.direction_linear = nn.Linear(self.hidden_size, self.hidden_size)

        # To produce final hidden representation
        self.linear1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)


    def forward(self, x):
        h = x.relu()
        has_child =  self.HASCHILD_linear(h) + x
        no_child = self.NOCHILD_linear(h) + x
        h = torch.cat([no_child.unsqueeze(-2), has_child.unsqueeze(-2)], dim=-2)
        x = x.unsqueeze(-2).expand(*h.shape)
        h = self.valence_linear(h.relu()).relu()

        left_h = self.LEFT_linear(h)  + x
        right_h = self.RIGHT_linear(h) + x
        h = torch.cat([left_h.unsqueeze(-3), right_h.unsqueeze(-3)], dim=-3)

        h = self.direction_linear(h.relu()).relu()

        return self.linear2(self.linear1(h)).relu()


