"""
https://github.com/SahinLale/StochasticMirrorDescent/blob/master/SMD_opt.py

стохастический зеркальный спуск
"""
import torch
from torch.optim import Optimizer
    

class SMD_qnorm(Optimizer):

    def __init__(self, params, lr=0.01, q=3):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 2.0 <= q:
            raise ValueError("Invalid q_norm value: {}".format(q))

        defaults = dict(lr=lr, q=q)
        super(SMD_qnorm, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(SMD_qnorm, self).__setstate__(state)
    
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                # q norm potential function
                update = (group['q'])* (torch.abs(p.data)**(group['q']-1)) * torch.sign(p.data) - group['lr'] * d_p
                p.data = (torch.abs(update/(group['q']))**(1/(group['q'] - 1))) * torch.sign(update) 
