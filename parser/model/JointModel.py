import torch.nn as nn
import torch
import sys
from parser.model import NDMV, SiblingNDMV, LNDMV



class JointFirstSecondOrderModel(nn.Module):
    '''
    Agreement-based training for joint First and Second-Order model.
    Please set First-Order model as model1
    and Second-Order model as model2
    '''
    def __init__(self, args, dataset):
        super(JointFirstSecondOrderModel, self).__init__()
        self.device = dataset.device
        model1_arg = args.model1
        model2_arg = args.model2
        self.model1 = self._get_model(model1_arg, dataset)
        self.model2 = self._get_model(model2_arg, dataset)


    def _get_model(self, hparams, dataset):
        if hparams.model_name == 'NDMV':
            model = NDMV(hparams, dataset)

        elif hparams.model_name == 'SiblingNDMV':
            model = SiblingNDMV(hparams, dataset)

        elif hparams.model_name == 'LexicalizedNDMV':
            model = LNDMV(hparams, dataset)

        else:
            raise NameError

        return model

    def forward(self, x):
        model1o = self.model1.forward(x)
        model2o = self.model2.forward(x)

        attach = model1o['attach'].unsqueeze(-2) + model2o['attach']
        decision = model1o['decision'].unsqueeze(2) + model2o['decision']
        root = model1o['root'] + model2o['root']
        kl = model1o['kl'] + model2o['kl']

        return {'attach': attach,
                'decision': decision,
                'root': root,
                'kl': kl}








