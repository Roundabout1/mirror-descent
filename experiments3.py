from torch.utils.data import DataLoader
from FCnet import FCnet
from loss import Loss_L2, Loss
from main import concat_results, make_dataloaders, make_models, run_tests, setup_MNIST, device, EXP_ROOT 
from tester import NetTester
from SMD_opt import SMD_qnorm
from SMD_opt2 import SMD_qnorm2
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torchvision import datasets, transforms
train_dataset, test_dataset = setup_MNIST()
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=True)

num_method = 2
full_scale = 60000
full_scale_epochs = 5
train_sizes = [full_scale]
train_batches = [256]
dont_skips = [full_scale_epochs]
test_every = [1]
root_folder = os.path.join(EXP_ROOT, "experiments3")
super_model = FCnet(800, 28*28, 10, 2).to(device)
for parameter_i in range(len(train_sizes)):
    cur_root_folder = os.path.join(root_folder, f'iteration_{parameter_i}')
    os.makedirs(cur_root_folder, exist_ok=True)
    data_folder = os.path.join(cur_root_folder, 'train_data')
    os.makedirs(data_folder, exist_ok=True)
    train_dataloaders = make_dataloaders(4, train_dataset, train_sizes[parameter_i], train_batches[parameter_i], 10, train_sizes[parameter_i] == full_scale)
    for T_DL in range(len(train_dataloaders)):
        torch.save(train_dataloaders[T_DL], os.path.join(data_folder, f'dataset_{T_DL}.pt'))
    for method_i in range(num_method):
        if method_i == 0:
            destination = os.path.join(cur_root_folder, 'SMD1')
        else:
            destination = os.path.join(cur_root_folder, 'SMD2')
        # зеркальный спуск
        models = make_models(len(train_dataloaders), super_model)
        # создаём тестеры
        num_testers = len(models)
        testers = []
        for tester_i in range(num_testers):
            if method_i == 0:
                optimizer = SMD_qnorm(models[tester_i].parameters(), lr=0.05, q=3)
            else:
                optimizer = SMD_qnorm2(models[tester_i].parameters(), lr=0.05, q=10)
            cur_tester = NetTester(
            model=models[tester_i],
            device=device,
            train_dataloader=train_dataloaders[tester_i],
            test_dataloader=test_dataloader,
            optimizer=optimizer,
            loss=Loss(nn.CrossEntropyLoss()),
            )
            testers.append(cur_tester)
        run_tests(testers, full_scale_epochs*(full_scale//train_sizes[parameter_i]), dont_skips[parameter_i], test_every[parameter_i])
        for tester in testers:
            tester.save_results(destination, 'SMD', True)
        main_tester = concat_results(testers)
        main_tester.save_results(destination, 'average', True)