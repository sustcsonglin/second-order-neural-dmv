import torch
import torch.nn as nn

class FactorizedBilinear(nn.Module):
    def __init__(self, in_size, r):
        super(FactorizedBilinear, self).__init__()
        self.in_size = in_size
        self.r = r
        self.project1 = nn.Linear(self.in_size, self.r)
        self.project2 = nn.Linear(self.in_size, self.r)

    def forward(self, x1, x2):
        x1 = self.project1(x1)
        x2 = self.project2(x2)
        if len(x1.shape) == 5:
            return torch.einsum("bhdve, bcdve -> bhcdv", x1, x2)
        elif len(x1.shape) == 4:
            return torch.einsum("hdve, cdve -> hcdv", x1, x2)
        else:
            raise NotImplementedError






