import torch
def weight_norm(model, l2_lambda, ignore_bias=False):
    norm = 0

    for param in model.parameters():
        if not ignore_bias or len(param.shape) != 1:
            norm += torch.sum(param**2)
    
    norm = torch.sqrt(norm)
    return norm*l2_lambda