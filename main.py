import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from FCnet import FCnet
from tester import NetTester
from data_load import balancing
from customSGD import CustomSGD
from loss import Loss, Loss_L2
from SMD_opt import SMD_qnorm
OUTPUT_ROOT = "output"

def get_device():
    device = (
        "cuda"
        if torch.cuda.is_available()
            else "mps"
        if torch.backends.mps.is_available()
            else "cpu"
    )
    print(f"Using {device} device")
    return device

device = get_device()

def make_models(num_models, model):
    """
    создать num_models одинаковых моделей
    """
    models = []
    for i in range(num_models):
        cur_model = model.clone().to(device)
        models.append(cur_model)
    return models

def make_dataloaders(num_loaders, dataset, dataloader_size, batch_size, lables_num=10):
    """
    создать num_loaders случайных выборок (DataLoader) из dataset
    """
    dataloaders = []
    for i in range(num_loaders):
        if dataloader_size is not None:
            balanced_dataset = balancing(dataset, lables_num, dataloader_size)
        else:
            balanced_dataset = dataset
        dataloader = DataLoader(balanced_dataset, batch_size=batch_size, shuffle=True)
        dataloaders.append(dataloader)
    return dataloaders

def make_testers(models, dataloaders, tester):
    """
    запустить len(models) тестеров, созданных по подобию tester,
    len(modeles) должно равняться len(dataloaders)
    """
    num_testers = len(models)
    if len(dataloaders) != num_testers:
        return None
    
    testers = []
    for i in range(num_testers):
        cur_tester = tester.clone()
        cur_tester.train_dataloader = dataloaders[i]
        cur_tester.model = models[i]
        testers.append(cur_tester)
    return testers
def run_tests(testers, epochs, dont_skip=-1, test_every=1):
    """
    запустить тестеры 
    
    обучение модели и получение результатов обучения на тестовой и обучающих выборках
    
    dont_skip - до какой эпохи не пропускать тесты (значение меньше нуля будет означать не пропускать тесты)
    
    test_every - тестировать каждую test_every эпоху, все остальноё - пропустить
    """
    for i in testers:
        i.train(epochs, dont_skip, test_every)
def concat_results(testers):
    """
    Получить средние значения на тестовых и обучающих выборках для testers

    ВНИМАНИЕ! Все характеристики, кроме значений точности и функции потерь берутся из первого тестера
    """
    num_testers = len(testers)
    if num_testers < 1:
        return None
    main_tester = testers[0]
    num_tests = len(main_tester.test_results)
    for i in range(num_testers):
        print(main_tester.test_results[i])
        print(main_tester.train_results[i])
    for i in range(num_testers):
        for j in  range(num_tests):
            print(testers[i].train_results[j])
    for i in range(num_testers):
        for j in  range(num_tests):
            print(testers[i].test_results[j])

    for i in range(1, num_testers):
        for j in  range(num_tests):
            main_tester.train_results[j] = (main_tester.train_results[j][0], 
                                            main_tester.train_results[j][1] + testers[i].train_results[j][1], 
                                            main_tester.train_results[j][2] + testers[i].train_results[j][2])
            main_tester.test_results[j] = (main_tester.test_results[j][0],
                                            main_tester.test_results[j][1] + testers[i].test_results[j][1], 
                                            main_tester.test_results[j][2] + testers[i].test_results[j][2])
    for i in range(num_tests):
        main_tester.train_results[i] = (main_tester.train_results[i][0],
                                            main_tester.train_results[i][1] / num_testers, 
                                            main_tester.train_results[i][2] / num_testers)
        main_tester.test_results[i] =  (main_tester.test_results[i][0],
                                            main_tester.test_results[i][1] / num_testers, 
                                            main_tester.test_results[i][2] / num_testers)
    return main_tester
def setup():
    """
    подготовка к тестированию и обучению моделей
    """
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
    return train_dataset, test_dataset

train_dataset, test_dataset = setup()
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)
train_dataloaders = make_dataloaders(4, train_dataset, 1000, 100, 10)
super_model = FCnet(800, 28*28, 10, 2).to(device)
models = make_models(len(train_dataloaders), super_model)
# создаём тестеры
num_testers = len(models)
testers = []
for i in range(num_testers):
    cur_tester = NetTester(
      model=models[i],
      device=device,
      train_dataloader=train_dataloaders[i],
      test_dataloader=test_dataloader,
      optimizer=optim.SGD(models[i].parameters(), lr=0.05),
      loss=Loss_L2(loss_fn=nn.CrossEntropyLoss(), model_parameters=models[i].parameters(), l2_lambda=0.01),
      show_progress=True
    )
    testers.append(cur_tester)
run_tests(testers, 20, 5, 2)
main_tester = concat_results(testers)
main_tester.save_results(OUTPUT_ROOT, 'test')