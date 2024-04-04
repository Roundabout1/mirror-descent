import torch
class Loss():
    def __init__(self, loss_fn):
        self.loss_fn = loss_fn
    def apply(self, pred, y):
        return self.loss_fn(pred, y)
    def __str__(self) -> str:
        return str(self.loss_fn)

class Loss_L2(Loss):
    def __init__(self, loss_fn, model_parameters, l2_lambda, ignore_bias=False):
        super().__init__(loss_fn)
        self.model_parameters = model_parameters
        self.l2_lambda = l2_lambda
        self.ignore_bias = ignore_bias
    def apply(self, pred, y):
        loss = super().apply(pred, y)
        norm = 0
        for param in self.model_parameters:
            if not self.ignore_bias or len(param.shape) != 1:
                norm += torch.sum(param**2)
        return loss + norm*self.l2_lambda
    def __str__(self) -> str:
        return '{loss_function} with L2-Regularization = {L2}, ignore bias = {bias}'.format(loss_function=super().__str__(), L2=self.l2_lambda, bias=self.ignore_bias)