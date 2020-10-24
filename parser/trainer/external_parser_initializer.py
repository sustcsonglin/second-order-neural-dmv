import torch
from parser.helper.loader_wrapper import LoaderWrapper
class External_parser_initializer:
    def __init__(self, device):
        self.device = device

    def initialize(self, model, loader, optimizer, hparams):
        model.train()
        for epoch in range(hparams.max_epoch):
            loader_autodevice = LoaderWrapper(loader, device=self.device)
            for x, y in loader_autodevice:
                optimizer.zero_grad()
                rules = model(x)
                loss = torch.sum(rules['attach'] * y['attach']) + \
                       torch.sum(rules['decision'] * y['decision']) + \
                       torch.sum(rules['root'] * y['root'])
                loss = -loss / rules['attach'].shape[0]
                loss.backward()
                optimizer.step()

            print("Initialization epoch:{} finied.".format(epoch))



