import torch

'''
Automatically put the tensor into the given device.
'''
class LoaderWrapper():
    def __init__(self, loader, device):
        self.loader = loader
        self.loader_iter = iter(loader)
        self.device = device

    def __iter__(self):
        return self

    def __next__(self):
        batch_x, batch_y = next(self.loader_iter)
        for name, key in batch_x.items():
            if type(key) is torch.Tensor:
                batch_x[name] = key.to(self.device)
        for name, key in batch_y.items():
            if type(key) is torch.Tensor:
                batch_y[name] = key.to(self.device)
        return batch_x, batch_y

    def __getitem__(self, item):
        return self.loader[item]


    def __len__(self):
        return len(self.loader)




