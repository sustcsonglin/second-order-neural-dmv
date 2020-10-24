import torch
import torch.nn as nn


class FactorizedTrilinear(nn.Module):
    def __init__(self, in_size, r):
        super(FactorizedTrilinear, self).__init__()
        self.in_size = in_size
        self.r = r
        self.project1 = nn.Linear(self.in_size, self.r)
        self.project2 = nn.Linear(self.in_size, self.r)
        self.project3 = nn.Linear(self.in_size, self.r)



    def forward(self, x1, x2, x3):
        '''
        :param x1: head:
        :param x2: sibling
        :param x3: child
        :return: attach score
        '''
        x1 = self.project1(x1)
        x2 = self.project2(x2)
        x3 = self.project3(x3)
        return torch.einsum("bzdve, bxdve, bcdve -> bzxcdv", x1, x2, x3)







