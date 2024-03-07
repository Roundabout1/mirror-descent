import torch
def weight_norm(model):
    norm = 0

    for param in model.parameters():
        norm += torch.sum(param**2)
    
    norm = torch.sqrt(norm)
    return norm