import torch
import torch.nn as nn
from parser.modules.res import ResLayer
from parser.modules.factorized_bilinear import FactorizedBilinear
from parser.modules.skip_connect_encoder import SkipConnectEncoder
from parser.const import *


class LVDMV(nn.Module):
    def __init__(self, args, dataset):
        super(LVDMV, self).__init__()
        self.device = dataset.device
        self.hidden_states = args.num_states
        self.num_word = len(dataset.word_vocab)
        self.s_dim = args.state_emb_size
        self.hidden_size = self.s_dim
        self.attach_r = args.attach_r
        self.decision_r = args.decision_r
        self.root_r = args.root_r


        #emb
        self.emb_for_unary = nn.Parameter(torch.randn(self.hidden_states, self.s_dim))
        self.emb_for_binary = nn.Parameter(torch.randn(self.hidden_states, self.s_dim))
        self.root_emb = nn.Parameter(torch.randn(1, self.s_dim))
        self.decision_emb = nn.Parameter(torch.randn(2, self.hidden_size))


        #unary rule:
        self.vocab_mlp = nn.Sequential(nn.Linear(self.s_dim,self.s_dim),
                                       ResLayer(self.s_dim, self.s_dim),
                                       ResLayer(self.s_dim, self.s_dim),
                                       nn.Linear(self.s_dim, self.num_word))

        # root rule:
        self.root_mlp =  nn.Sequential(nn.Linear(self.s_dim,self.s_dim),
                                       ResLayer(self.s_dim, self.s_dim),
                                       ResLayer(self.s_dim, self.s_dim),
                                       nn.Linear(self.s_dim, self.hidden_states))


        self.encoder = SkipConnectEncoder(hidden_size=self.hidden_size)
        self.attach_scorer = FactorizedBilinear(in_size=self.hidden_size, r=self.attach_r)
        self.decision_scorer = FactorizedBilinear(in_size=self.hidden_size, r=self.decision_r)
        # self.root_scorer = FactorizedBilinear(in_size=self.hidden_size, r=self.root_r)

        self.parent_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.child_linear =  nn.Linear(self.hidden_size, self.hidden_size)
        # self.root_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.decision_linear = nn.Linear(self.hidden_size, self.hidden_size)








    def forward(self, x):

        x = x['word']
        b, n = x.shape
        s = self.s_dim

        emb = self.emb_for_binary
        h_parent = self.encoder(self.parent_linear(emb))
        h_child = self.encoder(self.child_linear(emb))

        # h_root = self.encoder(self.root_linear(self.root_emb)).unsqueeze(0).expand(b, 1, DIRECTION_NUM, VALENCE_NUM, s)

        h_decision = self.encoder(self.decision_linear(self.decision_emb))


        def unary():
            unary_prob = self.vocab_mlp(self.emb_for_unary).log_softmax(-1)
            unary_prob = unary_prob.unsqueeze(0).unsqueeze(1).expand(
                b, n, self.hidden_states, self.num_word
            )
            indices = x.unsqueeze(2).expand(b, n, self.hidden_states).unsqueeze(3)
            unary_prob = torch.gather(unary_prob, 3, indices).squeeze(3)
            return unary_prob


        def root():
            return self.root_mlp(self.root_emb).log_softmax(-1).expand(b, self.hidden_states)

        def decision():
            dec_prob = self.decision_scorer(h_parent, h_decision).permute(0, 2, 3, 1).log_softmax(-1)
            return dec_prob.unsqueeze(0).expand(b, *dec_prob.shape)

        def attach():
            attach_prob = self.attach_scorer(h_parent, h_child).log_softmax(1)
            return attach_prob.unsqueeze(0).expand(b, *attach_prob.shape)


        return {'unary': unary(),
                'root': root(),
                'decision': decision(),
                'attach': attach(),
                'kl': torch.tensor(0, device=self.device)
                }






