import torch
import torch.nn as nn
from parser.modules.factorized_bilinear import FactorizedBilinear
from parser.modules.skip_connect_encoder import SkipConnectEncoder
from parser.const import *

class LNDMV(nn.Module):
    def __init__(self, args, dataset):
        super(LNDMV, self).__init__()
        self.device = dataset.device
        self.pos_emb_size = args.pos_emb_size
        self.word_emb_size = args.word_emb_size
        self.hidden_size = self.pos_emb_size + self.word_emb_size

        self.dropout_rate = args.dropout_rate
        self.pos_states = len(dataset.pos_vocab)
        self.word_states = len(dataset.word_vocab)
        self.states = len(dataset.word_vocab)

        self.attach_r = args.attach_r
        self.decision_r = args.decision_r
        self.root_r = args.root_r

        self.encoder = SkipConnectEncoder(hidden_size=self.hidden_size)
        self.attach_scorer = FactorizedBilinear(in_size=self.hidden_size, r=self.attach_r)
        self.decision_scorer = FactorizedBilinear(in_size=self.hidden_size, r=self.decision_r)
        self.root_scorer = FactorizedBilinear(in_size=self.hidden_size, r=self.root_r)

        self.dropout = nn.Dropout(self.dropout_rate)

        self.parent_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.child_linear =  nn.Linear(self.hidden_size, self.hidden_size)
        self.root_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.decision_linear = nn.Linear(self.hidden_size, self.hidden_size)

        self.pos_emb = nn.Embedding(self.pos_states, self.pos_emb_size)
        self.word_emb = nn.Embedding(self.word_states, self.word_emb_size)
        self.root_emb = nn.Parameter(torch.rand(1, self.hidden_size))
        self.decision_emb = nn.Parameter(torch.randn(2, self.hidden_size))

        try:
            print("Using pretrained word-embedding")
            # self.word_emb.from_pretrained()
            self.word_emb = nn.Embedding.from_pretrained(dataset.pretrained_emb, freeze=False)
        except:
            pass

        self.word2pos = dataset.word2pos

    def forward(self, x):
        word = x['word']
        pos = x['pos_tag']
        b, n = word.shape
        word_emb = self.word_emb(word)
        pos_emb = self.pos_emb(pos)
        x = torch.cat([pos_emb, word_emb], dim=-1)

        x = self.dropout(x)

        s = x.shape[-1]
        h_parent = self.encoder(self.parent_linear(x))
        h_child = self.encoder(self.child_linear(self.child_emb)).unsqueeze(0).expand(b, self.states, DIRECTION_NUM, VALENCE_NUM, s)
        h_root = self.encoder(self.root_linear(self.root_emb)).unsqueeze(0).expand(b, 1, DIRECTION_NUM, VALENCE_NUM, s)
        h_decision = self.encoder(self.decision_linear(self.decision_emb)).unsqueeze(0).expand(b, 2, DIRECTION_NUM, VALENCE_NUM, s)

        #drop unk and pad.
        # word = (word-2).clamp_min(0)

        def attach():
            attach = self.attach_scorer(h_parent, h_child).log_softmax(2)
            target_size = torch.Size([b, n, n, 2, 2])
            attach = torch.gather(attach, 2, word.reshape(b, 1, n, 1, 1).expand(target_size))
            left_mask = torch.tril(torch.ones(n, n, device=self.device), diagonal=-1)
            right_mask = torch.triu(torch.ones(n, n, device=self.device), diagonal=1)
            attach = attach[..., LEFT, :] * left_mask.unsqueeze(0).unsqueeze(-1) +\
                     attach[..., RIGHT, :] * right_mask.unsqueeze(0).unsqueeze(-1)
            return attach

        def decision():
            decision = self.decision_scorer(h_parent, h_decision).permute(0, 1, 3, 4, 2).log_softmax(-1)
            return decision

        def root():
            root = self.root_scorer(h_root, h_child).sum([-1, -2]).log_softmax(-1).squeeze(1)
            return torch.gather(root, 1, word)

        return {'attach': attach(),
                'decision': decision(),
                'root': root(),
                'kl': torch.tensor(0, device=self.device)}

    @property
    def child_emb(self):
        # exclude pad and unk.
        return torch.cat([self.pos_emb(self.word2pos), self.word_emb.weight], dim=-1)


























