from parser.dmvs.dmv import DMV
import torch
from parser.const import *


'''
Implementation of sibling DMV. 
'''
class DMV2o_sib(DMV):
    def __init__(self, device):
        super(DMV2o_sib, self).__init__(device)

    '''
    attach:  (b, N, N, N, valence):  batch_size, parent position, sibling position, child position, valence
    alpha[A][C][L]:  (b, N, N, N, valence): batch, span starts position, span ends position, span's next child's position who is out of 
    the span (0 for no more child)
    '''
    @torch.enable_grad()
    def _inside(self, rules, lens=None, mbr=False, viterbi=False, get_expected_counts=False, decoding=False):
        attach_score = rules['attach']
        root_score = rules['root']
        decision_score = rules['decision']
        semiring_plus = self.get_plus_semiring(viterbi)
        b, N, *_ = attach_score.shape
        if lens is None:
            lens = torch.zeros(b, device=self.device,dtype=torch.long).fill_(N)
        N += 1
        attach = torch.zeros(b, N, N, N, VALENCE_NUM, device=self.device).fill_(self.huge)
        decision = torch.zeros(b, N, N, DIRECTION_NUM, VALENCE_NUM, DECISION_NUM, device=self.device).fill_(self.huge)
        alpha = [
            [
                [torch.zeros(b, N, N, N, VALENCE_NUM, device=self.device).fill_(self.huge) for _ in range(2)] for _ in range(2)
            ] for _ in range(2)
        ]
        # treat the root token as the first token of the sentence.
        attach[:, 0, 1:, 0, NOCHILD] = root_score
        attach[:, 1:, 1:, :, :] = attach_score
        decision[:, 0, RIGHT, :, :] = 0
        decision[:, 1:] = decision_score
        alpha[A][C][L][:, :, 0, ...] =  decision[:, :, :, LEFT, :, STOP]
        alpha[B][C][L][:, :, -1, ...] =  decision[:, :, :, LEFT, :, STOP]
        alpha[A][C][R][:, :, 0, ...] =  decision[:, :, :, RIGHT,:, STOP]
        alpha[B][C][R][:, :, -1, ...] =  decision[:, :, :, RIGHT,:, STOP]
        if decoding:
            arc_indicator = torch.zeros(b, N, N, device=self.device, requires_grad=True)
            attach = attach + arc_indicator.unsqueeze(-1).unsqueeze(-1)
        start_idx = 1
        for k in range(1, N-start_idx):
            g = torch.arange(N-k-start_idx)
            f = torch.arange(start_idx, N - k), torch.arange(k+start_idx, N)
            ACL = alpha[A][C][L][:, start_idx: N - k, :k]
            ACR = alpha[A][C][R][:,  start_idx: N - k, :k]
            BCL = alpha[B][C][L][:,  k+start_idx:, N - k:]
            BCR = alpha[B][C][R][:,  k+start_idx:, N - k :]
            x = semiring_plus(ACR[..., 0, NOCHILD] + BCL[:, g, :, f[0], HASCHILD].transpose(0, 1), dim=2)
            arcs_l = x[..., None, None] + attach[:, f[1], f[0], :, :] + decision[:, f[1], :, LEFT, :, GO]
            alpha[A][I][L][:,  start_idx:N-k, k] = arcs_l
            alpha[B][I][L][:, start_idx+k:N, N - k - 1] = arcs_l
            x = semiring_plus(ACR[..., g, :,  f[1], HASCHILD].transpose(0, 1) + BCL[..., 0, NOCHILD], dim=2)
            arcs_r = x[..., None, None] + attach[:, f[0], f[1], :, :] + decision[:, f[0], :, RIGHT, :, GO]
            alpha[A][I][R][:, start_idx: N - k, k] = arcs_r
            alpha[B][I][R][:, k+start_idx:N, N - k - 1] = arcs_r
            AIR = alpha[A][I][R][:, start_idx: N - k, 1 : k + 1]
            BIL = alpha[B][I][L][:, k+start_idx:, N - k - 1 : N - 1]
            new = semiring_plus(ACL[..., 0, NOCHILD, None, None] + BIL, dim=2)
            alpha[A][C][L][:, start_idx: N - k, k] = new
            alpha[B][C][L][:, k+start_idx:N, N - k - 1] = new
            new = semiring_plus(AIR + BCR[..., 0, NOCHILD, None, None], dim=2)
            alpha[A][C][R][:, start_idx: N - k, k] = new
            alpha[B][C][R][:, k+start_idx:N, N - k - 1] = new
        # dealing with the root.
        root_incomplete_span = alpha[A][C][L][:, 1, :N - 1, 0, NOCHILD] + attach[:, 0, 1:, 0, NOCHILD]
        for k in range(1, N):
            AIR = root_incomplete_span[:, :k]
            BCR = alpha[B][C][R][:, k, N - k:]
            alpha[A][C][R][:, 0, k, 0, NOCHILD] = semiring_plus(AIR + BCR[..., 0, NOCHILD], dim=1)
        logZ = torch.gather(alpha[A][C][R][:, 0, :, 0, NOCHILD], -1, lens.unsqueeze(-1))
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
        elif get_expected_counts:
            grad = torch.autograd.grad(logZ.sum(), [attach_score, decision_score, root_score])
            return {'attach_grad': grad[0],
                    'decision_grad': grad[1],
                    'root_grad': grad[2]}
        return {"partition": logZ}
