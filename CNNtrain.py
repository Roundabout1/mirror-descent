"""
обучение свёрточной нейронной сети
"""
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tester import NetTester
from CNNet import ConvNet
from main import device
from loss import Loss
from SMD_opt import SMD_qnorm
def setup_cnn():
    # среднее и стандартное отклонение MNIST датасета равны 0.1307 и 0.3081
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) 

    # загрузка датасетов
    train_dataset = datasets.MNIST(root='./data', train=True, transform=trans, download=True) 
    test_dataset = datasets.MNIST(root='./data', train=False, transform=trans)
    return train_dataset, test_dataset

num_epochs = 5 
num_classes = 10 
batch_size = 100 
learning_rate = 0.01

# среднее и стандартное отклонение MNIST датасета равны 0.1307 и 0.3081
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) 

# загрузка датасетов
train_dataset = datasets.MNIST(root='./data', train=True, transform=trans, download=True) 
test_dataset = datasets.MNIST(root='./data', train=False, transform=trans)

# загрузчики
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=True) 
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model = ConvNet()
model.to(device)
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = SMD_qnorm(model.parameters(), lr=learning_rate)

tester = NetTester(
    model=model,
    device=device,
    train_dataloader=train_loader,
    test_dataloader=test_loader,
    optimizer= optimizer,
    loss=Loss(criterion)
)
tester.train(
    epochs=num_epochs
)
tester.save_results(output_root="CNN", minimum=True)