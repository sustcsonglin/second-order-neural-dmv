import torch
import torch.nn as nn

class VanillaDMV(nn.Module):
    def __init__(self,  dataset):
        super(VanillaDMV, self).__init__()
        self.device = dataset.device
        self.states = len(dataset.pos_vocab)
        self.root_param = nn.Parameter(torch.rand(self.states))
        self.attach_param = nn.Parameter(torch.rand(self.states, self.states, 2, 2))
        self.decision_param = nn.Parameter(torch.rand(self.states, 2 , 2, 2))

    def _initialize(self, params):
        self.root_param.data.copy_(params['root'])
        self.attach_param.data.copy_(params['attach'])
        self.decision_param.data.copy_(params['decision'])


    def forward(self, input):
        x = input['pos_tag'].to(self.device)
        b, n = x.shape
        x = x.contiguous()

        def attach():
            target_size1 = torch.Size([b, n, self.states, 2, 2])
            child_rule_prob = self.attach_param.unsqueeze(0).expand(b, *self.attach_param.shape)
            child_rule_prob = torch.gather(child_rule_prob, 1, x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(target_size1))
            target_size2 = torch.Size([b, n, n, 2, 2])
            child_rule_prob = torch.gather(child_rule_prob, 2, x.unsqueeze(-2).unsqueeze(-1).unsqueeze(-1).expand(target_size2))
            left_mask = torch.tril(torch.ones(n, n, device=self.device), diagonal=-1)
            right_mask = torch.triu(torch.ones(n, n, device=self.device), diagonal=1)
            LEFT = 0
            RIGHT = 1
            return child_rule_prob[:, :, :, LEFT, :] * left_mask.unsqueeze(0).unsqueeze(-1) +\
                   child_rule_prob[:, :, :, RIGHT, :] * right_mask.unsqueeze(0).unsqueeze(-1)

        def decision():
            target_size = torch.Size([b, n, 2, 2, 2])
            return torch.gather(self.decision_param.log_softmax(-1).unsqueeze(0).expand(b, *self.decision_param.shape),  1,
                                x.reshape(b, n, 1, 1, 1).expand(target_size))

        def root():
            return torch.gather(self.root_param.log_softmax(-1).unsqueeze(0).expand(b, self.states),
                                1, x)

        return {'attach': attach(),
                'decision': decision(),
                'root': root()}




