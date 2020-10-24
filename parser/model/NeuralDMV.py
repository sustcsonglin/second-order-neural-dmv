import torch
import torch.nn as nn
from parser.modules.factorized_bilinear import FactorizedBilinear
from parser.modules.skip_connect_encoder import SkipConnectEncoder
from parser.const import *

'''
Self-implementation of : 
Yong Jiang, Wenjuan Han, and Kewei Tu, "Unsupervised Neural Dependency Parsing".
'''

class NDMV(nn.Module):
    def __init__(self, args, dataset):

        super(NDMV, self).__init__()
        self.device = dataset.device
        self.num_pos = len(dataset.pos_vocab)

        self.pos_emb_size = args.pos_emb_size
        self.hidden_size = args.hidden_size
        self.pre_child_size = args.pre_child_size
        self.pre_decision_size = args.pre_decision_size
        self.pos_emb = nn.Embedding(self.num_pos, self.pos_emb_size)
        self.val_emb = nn.Parameter(torch.randn(2, self.pos_emb_size))
        self.root_emb = nn.Parameter(torch.randn(1, self.pos_emb_size))
        self.activate = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout_rate)
        self.shared_linear_left = nn.Linear(2*self.pos_emb_size, self.hidden_size)
        self.shared_linear_right = nn.Linear(2*self.pos_emb_size, self.hidden_size)
        self.child_linear = nn.Linear(self.hidden_size, self.pre_child_size)
        self.decision_linear = nn.Linear(self.hidden_size, self.pre_decision_size)
        self.child_out_linear = nn.Linear(self.pre_child_size, self.num_pos)
        self.decision_out_linear = nn.Linear(self.pre_decision_size, 2)
        self.root_linear =   nn.Linear(self.pos_emb_size, self.num_pos)



    def forward(self, input):
        x = input['pos_tag'].to(self.device)
        b_size, n = x.shape
        emb = self.pos_emb(x)
        val = self.val_emb
        target_size = torch.Size([b_size, n, 2, self.pos_emb_size])
        emb = torch.cat([
            emb.unsqueeze(-2).expand(target_size),
            val.unsqueeze(0).unsqueeze(0).expand(target_size)
        ], dim=-1)
        emb = self.dropout(emb)
        h1 = self.shared_linear_left(emb)
        h2 = self.shared_linear_right(emb)
        h = self.activate(torch.cat([h1.unsqueeze(-2), h2.unsqueeze(-2)], dim=-2))
        h = self.dropout(h)
        def attach():
            h1 = self.child_linear(h)
            h1 = self.activate(h1)
            t_size = torch.Size([b_size, n, 2, 2, n])
            child_rule_prob = torch.gather(self.child_out_linear(h1).log_softmax(-1), -1,  x.reshape(b_size, 1, 1, 1, n).expand(t_size))
            child_rule_prob = child_rule_prob.permute(0, 1, 4, 2, 3)
            left_mask = torch.tril(torch.ones(n, n, device=self.device), diagonal=-1)
            right_mask = torch.triu(torch.ones(n, n, device=self.device), diagonal=1)
            return child_rule_prob[:, :, :, LEFT, :] * left_mask.unsqueeze(0).unsqueeze(-1) +\
                   child_rule_prob[:, :, :, RIGHT, :] * right_mask.unsqueeze(0).unsqueeze(-1)

        def decision():
            h1 = self.decision_linear(h)
            h1 = self.activate(h1)
            out = self.decision_out_linear(h1).log_softmax(-1)
            return out

        def root():
            return torch.gather(
                self.root_linear(self.root_emb).log_softmax(-1).expand(b_size, self.num_pos),
                -1,
                x
            )

        return {'attach': attach(),
                'decision': decision(),
                'root': root(),
                'kl': torch.tensor(0, device=self.device)}


