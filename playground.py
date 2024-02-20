from FCnet import FCnet
from customSGD import CustomSGD

model = FCnet(100, 28*28, 10, 2)

for p in model.parameters():
    if len(p.shape) != 1:
        print(p.shape) 