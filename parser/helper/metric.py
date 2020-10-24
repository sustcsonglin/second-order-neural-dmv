# -*- coding: utf-8 -*-

import torch


class Metric(object):

    def __lt__(self, other):
        return self.score < other

    def __le__(self, other):
        return self.score <= other

    def __ge__(self, other):
        return self.score >= other

    def __gt__(self, other):
        return self.score > other

    @property
    def score(self):
        return -1e9


class UAS(Metric):
    def __init__(self, eps=1e-8):
        self.eps = eps
        self.total = 0.0
        self.correct = 0.0

    def __call__(self, predict, gold, lens):
        self.correct +=  (predict == gold).sum().detach()
        self.total += lens.sum().detach()

    @property
    def avg_uas(self):
        return self.correct/self.total

    @property
    def score(self):
        return self.avg_uas.item()

    def __repr__(self):
        return "uas: {}, correct:{} ,total:{}".format(self.score, self.correct, self.total)









class LossMetric(Metric):
    def __init__(self, eps=1e-8):
        super(Metric, self).__init__()
        self.eps = eps
        self.total = 0.0
        self.total_likelihood = 0.0
        self.total_kl = 0.0
        self.calling_time = 0


    def __call__(self, likelihood, kl):
        self.calling_time += 1
        self.total += likelihood.shape[0]
        self.total_likelihood += likelihood.detach_().sum()
        self.total_kl += kl.detach_().sum()

    @property
    def avg_likelihood(self):
        return self.total_likelihood / self.total

    @property
    def avg_kl(self):
        return self.total_kl/self.total

    def __repr__(self):
        return "avg likelihood: {} kl: {}, total likelihood:{}, n:{}".format(self.avg_likelihood,self.avg_kl,self.total_likelihood, self.total)

    def sync(self):
        import numpy as np
        total = torch.from_numpy(np.array(self.total)).float().type_as(self.total_likelihood)
        torch.distributed.reduce(total, 0, torch.distributed.ReduceOp.SUM)
        torch.distributed.reduce(self.total_likelihood, 0, torch.distributed.ReduceOp.SUM)
        torch.distributed.reduce(self.total_kl, 0, torch.distributed.ReduceOp.SUM)
        self.total = total.item()
    @property
    def score(self):
        return (self.avg_likelihood + self.avg_kl).item()



class LikelihoodMetric(Metric):
    def __init__(self, eps=1e-8):
        super(Metric, self).__init__()
        self.eps = eps
        self.total = 0.0
        self.total_likelihood = 0.0

    @property
    def score(self):
        return self.avg_likelihood


    def __call__(self, likelihood):
        self.total += likelihood.shape[0]
        self.total_likelihood += likelihood.detach_().sum()

    @property
    def avg_likelihood(self):
        return self.total_likelihood / self.total





    def __repr__(self):
        return "avg likelihood: {}".format(self.avg_likelihood)



