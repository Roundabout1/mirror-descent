"""
SMD без умножения на q
"""
# https://github.com/SahinLale/StochasticMirrorDescent/blob/master/SMD_opt.py
# стохастический зеркальный спуск 
import torch
from torch.optim import Optimizer
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors, \
    _take_tensors
    

class SMD_qnorm2(Optimizer):

    def __init__(self, params, lr=0.01, q =3):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 2.0 <= q:
            raise ValueError("Invalid q_norm value: {}".format(q))

        defaults = dict(lr=lr, q = q)
        super(SMD_qnorm2, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(SMD_qnorm2, self).__setstate__(state)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
    #           q norm potential function (без умножения на q)
                update = (torch.abs(p.data)**(group['q']-1)) * torch.sign(p.data) - group['lr'] * d_p
                p.data = (torch.abs(update/(group['q']))**(1/(group['q'] - 1))) * torch.sign(update)

        return loss 
    

