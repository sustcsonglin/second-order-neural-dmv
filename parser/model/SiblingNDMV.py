import torch
import torch.nn as nn
from parser.modules.skip_connect_encoder import SkipConnectEncoder
from parser.modules.factorized_trilinear import FactorizedTrilinear
from parser.modules.factorized_bilinear import FactorizedBilinear
from parser.const import *

class SiblingNDMV(nn.Module):
    def __init__(self, args, dataset):
        super(SiblingNDMV, self).__init__()
        self.device = dataset.device
        self.pos_emb_size = args.pos_emb_size
        self.hidden_size = args.hidden_size
        self.attach_r = args.attach_r
        self.decision_r = args.decision_r

        self.root_r = args.root_r
        self.states = len(dataset.pos_vocab)

        self.parent_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.sibling_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.child_linear =  nn.Linear(self.hidden_size, self.hidden_size)
        self.root_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.decision_linear = nn.Linear(self.hidden_size, self.hidden_size)

        self.encoder = SkipConnectEncoder(self.hidden_size)
        self.attach_scorer =  FactorizedTrilinear(self.hidden_size, self.attach_r)
        self.decision_scorer =  FactorizedTrilinear(self.hidden_size, self.decision_r)
        self.root_scorer = FactorizedBilinear(self.hidden_size, self.root_r)

        # self.drop_out = nn.Dropout(args.dropout_rate)

        # +1 for a special symbol NULL to represent no sibling.
        self.pos_emb = nn.Embedding(self.states + 1, self.pos_emb_size)
        self.root_emb = nn.Parameter(torch.rand(1, self.pos_emb_size))
        self.decision_emb = nn.Parameter(torch.randn(2, self.pos_emb_size))


    def forward(self, x):
        sent = x['pos_tag']
        x = self.pos_emb(sent)
        # x = self.drop_out(x)

        # exclude pad
        sent = (sent - 1).clamp_min(0)
        b, n, s = x.shape
        h_parent = self.encoder(self.parent_linear(x))
        x_sib = self.sibling_linear(x)
        x_sib = torch.cat([self.no_sib_emb.unsqueeze(0).unsqueeze(0).expand(b, 1, s), x_sib], dim=1)
        h_sib = self.encoder(x_sib)
        h_child = self.encoder(self.child_linear(self.child_emb)).unsqueeze(0).expand(b, self.states-1, DIRECTION_NUM, VALENCE_NUM, s)
        h_root = self.encoder(self.root_linear(self.root_emb)).unsqueeze(0).expand(b, 1, DIRECTION_NUM, VALENCE_NUM, s)
        h_decision = self.encoder(self.decision_linear(self.decision_emb)).unsqueeze(0).expand(b, 2, DIRECTION_NUM, VALENCE_NUM, s)

        def attach():
            attach = self.attach_scorer(h_parent, h_child, h_sib).log_softmax(2)
            target_size = torch.Size([b, n, n, n+1,  2, 2])
            attach = torch.gather(attach, 2, sent.reshape(b, 1, n, 1, 1, 1).expand(target_size))
            left_mask = torch.tril(torch.ones(n, n, device=self.device), diagonal=-1)
            right_mask = torch.triu(torch.ones(n, n, device=self.device), diagonal=1)
            attach = attach[..., LEFT, :] * left_mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) +\
                     attach[..., RIGHT, :] * right_mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            return attach


        def decision():
            decision = self.decision_scorer(h_parent, h_sib, h_decision).permute(0, 1, 2, 4, 5, 3).log_softmax(-1)
            return decision

        def root():
            root = self.root_scorer(h_root, h_child).sum([-1, -2]).log_softmax(-1).squeeze(1)
            root = torch.gather(root, 1, sent)
            return root


        return {'attach': attach(),
                'decision': decision(),
                'root': root(),
                'kl': torch.tensor(0., device=self.device)}

    @property
    def no_sib_emb(self):
        return self.pos_emb.weight[-1]

    @property
    def child_emb(self):
        # exclude pad
        return self.pos_emb.weight[1:-1]
