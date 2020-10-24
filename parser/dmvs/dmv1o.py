from parser.dmvs.dmv import DMV
import torch
from parser.const import *

class DMV1o(DMV):
    def __init__(self, device):
        super(DMV1o, self).__init__(device)

    @torch.enable_grad()
    def _inside(self, rules, lens=None, viterbi=False, mbr=False, get_expected_counts=False, decoding=False):
        '''
        :param rules: grammar rules.
        :param lens: sentence length
        :param viterbi: use viterbi algorithm
        :param mbr: use minimal bayes risk decoding algorithm
        :param get_expected_counts: used in EM algorithm to get expected counts of each grammar rules.
        :return:
        '''
        attach_score = rules['attach']
        root_score = rules['root']
        decision_score = rules['decision']
        semiring_plus = self.get_plus_semiring(viterbi)
        b, N, *_ = attach_score.shape
        if lens is None:
            lens = torch.zeros(b, device=self.device,dtype=torch.long).fill_(N)
        else:
            lens = lens.to(self.device)
        N += 1
        attach = torch.zeros(b, N, N, VALENCE_NUM, device=self.device).fill_(self.huge)
        decision = torch.zeros(b, N, DIRECTION_NUM, VALENCE_NUM, DECISION_NUM, device=self.device).fill_(self.huge)
        alpha = [
            [
                [torch.zeros(b, N, N, VALENCE_NUM, device=self.device).fill_(self.huge) for _ in range(2)] for _ in range(2)
            ] for _ in range(2)
        ]
        # treat the root token as the first token of the sentence.
        attach[:, 0, 1:, NOCHILD] = root_score
        attach[:, 1:, 1:, :] = attach_score
        decision[:, 0, RIGHT, :, :] = 0
        decision[:, 1:] = decision_score
        alpha[A][C][L][:, :, 0, :] =  decision[:, :,  LEFT, :, STOP]
        alpha[B][C][L][:, :, -1, :] =  decision[:, :, LEFT, :, STOP]
        alpha[A][C][R][:, :, 0, :] =  decision[:, :, RIGHT,:, STOP]
        alpha[B][C][R][:, :, -1, :] =  decision[:, :, RIGHT,:, STOP]
        if decoding:
            arc_indicator = torch.zeros(b, N, N, device=self.device, requires_grad=True)
            attach = attach + arc_indicator.unsqueeze(-1)
        for k in range(1, N):
            f = torch.arange(N - k), torch.arange(k, N)
            ACL = alpha[A][C][L][:, : N - k, :k]
            ACR = alpha[A][C][R][:,  : N - k, :k]
            BCL = alpha[B][C][L][:,  k:, N - k:]
            BCR = alpha[B][C][R][:,  k:, N - k :]
            x = semiring_plus(ACR[...,NOCHILD, None] + BCL[...,HASCHILD, None], dim=2)
            arcs_l = x + attach[:, f[1], f[0], :] + decision[:, f[1], LEFT, :, GO]
            alpha[A][I][L][:,  : N - k, k] = arcs_l
            alpha[B][I][L][:, k:N, N - k - 1] = arcs_l
            x = semiring_plus(ACR[...,HASCHILD, None] + BCL[..., NOCHILD, None], dim=2)
            arcs_r = x + attach[:, f[0], f[1], :] + decision[:, f[0], RIGHT, :, GO]
            alpha[A][I][R][:, : N - k, k] = arcs_r
            alpha[B][I][R][:, k:N, N - k - 1] = arcs_r
            AIR = alpha[A][I][R][:, : N - k, 1 : k + 1]
            BIL = alpha[B][I][L][:, k:, N - k - 1 : N - 1]
            new = semiring_plus(ACL[..., NOCHILD, None] + BIL, dim=2)
            alpha[A][C][L][:, : N - k, k] = new
            alpha[B][C][L][:, k:N, N - k - 1] = new
            new = semiring_plus(AIR + BCR[..., NOCHILD, None], dim=2)
            alpha[A][C][R][:, : N - k, k] = new
            alpha[B][C][R][:, k:N, N - k - 1] = new
        logZ = torch.gather(alpha[A][C][R][:, 0, :, NOCHILD], -1, lens.unsqueeze(-1))
        if get_expected_counts:
            grad = torch.autograd.grad(logZ.sum(), [attach_score, decision_score, root_score])
            return {'attach_grad': grad[0],
                    'decision_grad': grad[1],
                    'root_grad': grad[2]}
        if decoding:
            if viterbi:
                arc = torch.autograd.grad(logZ.sum(), arc_indicator)[0].nonzero()
                predicted = torch.zeros(b, N-1, device=self.device, dtype=torch.long).fill_(-1)
                predicted[arc[:, 0], arc[:, 2]-1] = arc[:, 1]
                return {'predicted_arc': predicted}
            elif mbr:
                arc_marginal = torch.autograd.grad(logZ.sum(), arc_indicator)[0]
                predicted = self._eisner(arc_marginal, lens)
                return {'predicted_arc': predicted}
            else:
                raise NotImplementedError
        return {'partition': logZ}
