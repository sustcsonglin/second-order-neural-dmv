import torch
from parser.const import *
'''
These implementations borrow the idea of "Torch-Struct: Deep Structured Prediction Library".
Ref: https://github.com/harvardnlp/pytorch-struct/blob/master/torch_struct/deptree.py
'''
class DMV():
    def __init__(self, device):
        self.device = device
        self.huge = -1e30

    def _inside(self, rules, lens, viterbi=False, mbr=False):
        raise NotImplementedError


    def get_plus_semiring(self, viterbi):
        if viterbi:
            def plus(x, dim):
                return torch.max(x, dim)[0]
        else:
            def plus(x, dim):
                return torch.logsumexp(x, dim)
        return plus


    @torch.no_grad()
    def _decode(self, rules, lens, viterbi=False, mbr=False):
        assert viterbi or mbr
        assert not (viterbi and mbr)
        return self._inside(rules, lens, viterbi=viterbi, mbr=mbr, decoding=True)


    @torch.enable_grad()
    def _eisner(self, attach, lens):
        '''
        :param attach: The marginal probabilities.
        :param lens: sentences lens
        :return: predicted_arcs
        '''
        b, N, *_ = attach.shape
        attach.requires_grad_(True)
        alpha = [
            [
                [torch.zeros(b, N, N, device=self.device).fill_(self.huge) for _ in range(2)] for _ in range(2)
            ] for _ in range(2)
        ]
        alpha[A][C][L][:, :, 0] = 0
        alpha[B][C][L][:, :, -1] =  0
        alpha[A][C][R][:, :, 0] =  0
        alpha[B][C][R][:, :, -1] = 0
        semiring_plus = self.get_plus_semiring(viterbi=True)
        # single root.
        start_idx = 1
        for k in range(1, N-start_idx):
            f = torch.arange(start_idx, N - k), torch.arange(k+start_idx, N)
            ACL = alpha[A][C][L][:, start_idx: N - k, :k]
            ACR = alpha[A][C][R][:,  start_idx: N - k, :k]
            BCL = alpha[B][C][L][:,  start_idx+k:, N - k:]
            BCR = alpha[B][C][R][:,  start_idx+k:, N - k :]
            x = semiring_plus(ACR + BCL, dim=2)
            arcs_l = x + attach[:, f[1], f[0]]
            alpha[A][I][L][:,  start_idx: N - k, k] = arcs_l
            alpha[B][I][L][:, k+start_idx:N, N - k - 1] = arcs_l
            x = semiring_plus(ACR + BCL, dim=2)
            arcs_r = x + attach[:, f[0], f[1]]
            alpha[A][I][R][:, start_idx: N - k, k] = arcs_r
            alpha[B][I][R][:, k+start_idx:N, N - k - 1] = arcs_r
            AIR = alpha[A][I][R][:, start_idx: N - k, 1 : k + 1]
            BIL = alpha[B][I][L][:, k+start_idx:, N - k - 1 : N - 1]
            new = semiring_plus(ACL + BIL, dim=2)
            alpha[A][C][L][:, start_idx: N - k, k] = new
            alpha[B][C][L][:, k+start_idx:N, N - k - 1] = new
            new = semiring_plus(AIR + BCR, dim=2)
            alpha[A][C][R][:, start_idx:N-k, k] = new
            alpha[B][C][R][:, start_idx+k:N, N - k - 1] = new
        # dealing with the root.
        root_incomplete_span = alpha[A][C][L][:, 1, :N-1] + attach[:, 0, 1:]
        for k in range(1,N):
            AIR = root_incomplete_span[:, :k]
            BCR = alpha[B][C][R][:, k, N-k:]
            alpha[A][C][R][:, 0, k] = semiring_plus(AIR+BCR, dim=1)
        logZ = torch.gather(alpha[A][C][R][:, 0, :], -1, lens.unsqueeze(-1))
        arc = torch.autograd.grad(logZ.sum(), attach)[0].nonzero()
        predicted = torch.zeros(b, N - 1, device=self.device, dtype=torch.long).fill_(-1)
        predicted[arc[:, 0], arc[:, 2] - 1] = arc[:, 1]
        return predicted













