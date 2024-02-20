import torch.nn as nn 
from torch.optim.optimizer import Optimizer
class CustomSGD(Optimizer):
    def __init__(self, params, lr=1e-3):
        super(CustomSGD, self).__init__(params, defaults={'lr': lr})
        for group in self.param_groups: 
            for p in group['params']:
                print(p.shape)


    def step(self): 
        for group in self.param_groups:
            for p in group['params']: 
                p.data -= group['lr'] * p.grad.data
                  