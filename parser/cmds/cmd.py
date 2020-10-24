# -*- coding: utf-8 -*-
import torch
from parser.helper.metric import  LossMetric, UAS
from tqdm import tqdm
import torch.nn as  nn

class CMD(object):

    def __call__(self, args):
        self.args = args


    @torch.no_grad()
    def evaluate(self, loader, model=None, dmv=None):
        if model == None:
            model = self.model
            dmv = self.dmv
        model.eval()
        metric = UAS()
        arg = self.args.test
        for x, y in loader:
            rules = model(x)
            if arg.decode == 'viterbi':
                ret = dmv._decode(rules, x['seq_len'], viterbi=True)
            elif arg.decode == 'mbr':
                ret = dmv._decode(rules, x['seq_len'], mbr=True)
            else:
                raise NotImplementedError
            metric(ret['predicted_arc'], y['head'], x['seq_len'])
        return metric

    def train(self, loader):
        self.model.train()
        metric = LossMetric()
        t = tqdm(loader, total=int(len(loader)),  position=0, leave=True)
        train_arg = self.args.train.training
        for x, _ in t:
            self.optimizer.zero_grad()
            rules = self.model(x)
            kl = rules['kl']
            ret = self.dmv._inside(rules, x['seq_len'])
            logZ = ret['partition']
            loss = (-logZ + kl).mean()
            loss.backward()
            if train_arg.clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                      train_arg.clip)
            self.optimizer.step()
            metric(logZ, kl)
            t.set_postfix(loss='%.2f' % metric.avg_likelihood % metric.avg_kl, refresh=False)
        return metric

    @torch.no_grad()
    def evaluate_likelihood(self, loader):
        self.model.eval()
        metric = LossMetric()
        for x, y in loader:
            rules = self.model(x)
            kl = rules['kl']
            ret = self.dmv._inside(rules, x['seq_len'])
            logZ = ret['partition']
            metric(logZ,kl)
        return metric