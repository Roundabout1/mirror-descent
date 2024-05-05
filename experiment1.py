from torch.utils.data import DataLoader
from FCnet import FCnet
from loss import Loss_L2, Loss
from main import concat_results, make_dataloaders, make_models, run_tests, setup, device
from tester import NetTester
from SMD_opt import SMD_qnorm
import torch.nn as nn
import torch.optim as optim

train_dataset, test_dataset = setup()
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=True)

full_scale = 60000
full_scale_epochs = 10
#train_sizes = [None, 30000, 20000, 10000, 5000, 1000, 500, 250, 100]
#train_batches = [256, 128, 96, 64, 32, 16, 8, 4, 2]
#dont_skips = [5, 10, 15, 20, 40, 50, 100, 150, 200]
#test_every = [2, 4, 6, 8, 16, 80, 160, 320]
train_sizes = [None, 30000, 20000]
train_batches = [256, 128, 96]
dont_skips = [5, 10, 15]
test_every = [2, 4, 6]
folder_name = "big_test1"
for parameter_i in range(len(train_sizes)):
    train_dataloaders = make_dataloaders(4, train_dataset, train_sizes[parameter_i], train_batches[parameter_i], 10)
    super_model = FCnet(800, 28*28, 10, 2).to(device)

    # градиентный спуск без регуляризации
    models = make_models(len(train_dataloaders), super_model)
    # создаём тестеры
    num_testers = len(models)
    testers = []
    for tester_i in range(num_testers):
        cur_tester = NetTester(
        model=models[tester_i],
        device=device,
        train_dataloader=train_dataloaders[tester_i],
        test_dataloader=test_dataloader,
        optimizer=optim.SGD(models[tester_i].parameters(), lr=0.05),
        loss=Loss(nn.CrossEntropyLoss()),
        )
        testers.append(cur_tester)
    run_tests(testers, full_scale_epochs*(full_scale//train_sizes[parameter_i]), dont_skips[parameter_i], test_every[parameter_i])
    main_tester = concat_results(testers)
    main_tester.save_results(folder_name, 'SGD')

    # градиентный спуск с регуляризацией
    models = make_models(len(train_dataloaders), super_model)
    # создаём тестеры
    num_testers = len(models)
    testers = []
    for tester_i in range(num_testers):
        cur_tester = NetTester(
        model=models[tester_i],
        device=device,
        train_dataloader=train_dataloaders[tester_i],
        test_dataloader=test_dataloader,
        optimizer=optim.SGD(models[tester_i].parameters(), lr=0.05),
        loss=Loss_L2(loss_fn=nn.CrossEntropyLoss(), model_parameters=models[tester_i].parameters(), l2_lambda=0.01),
        )
        testers.append(cur_tester)
    run_tests(testers, full_scale_epochs*(full_scale//train_sizes[parameter_i]), dont_skips[parameter_i], test_every[parameter_i])
    main_tester = concat_results(testers)
    main_tester.save_results(folder_name, 'SGDL2')

    # зеркальный спуск
    models = make_models(len(train_dataloaders), super_model)
    # создаём тестеры
    num_testers = len(models)
    testers = []
    for tester_i in range(num_testers):
        cur_tester = NetTester(
        model=models[tester_i],
        device=device,
        train_dataloader=train_dataloaders[tester_i],
        test_dataloader=test_dataloader,
        optimizer=SMD_qnorm(models[tester_i].parameters(), lr=0.05, q=3),
        loss=Loss(nn.CrossEntropyLoss()),
        )
        testers.append(cur_tester)
    run_tests(testers, full_scale_epochs*(full_scale//train_sizes[parameter_i]), dont_skips[parameter_i], test_every[parameter_i])
    main_tester = concat_results(testers)
    main_tester.save_results(folder_name, 'SMD')