import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from FCnet import FCnet
from tester import NetTester
from data_load import balancing

from customSGD import CustomSGD
OUTPUT_ROOT = "output"

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Загрузка данных MNIST
train_num = 60000
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

batch_size = 128

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

model = FCnet(800, 28*28, 10, 2).to(device)
tester = NetTester(
      model=model,
      device=device,
      train_dataloader=train_dataloader,
      test_dataloader=test_dataloader,
      optimizer=CustomSGD(model.parameters(), lr=0.05),
      loss_fn=nn.CrossEntropyLoss()
)

tester.train(epochs=5*(60000//train_num))

tester.save_results(OUTPUT_ROOT)