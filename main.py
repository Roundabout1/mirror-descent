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
"""
старый путь к результатам экспериментов
"""
OUTPUT_ROOT = "output"
"""
путь к папке с результатами экспериментов
"""
EXP_ROOT = "experiments"
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

def make_dataloaders(num_loaders, dataset, dataloader_size, batch_size, lables_num=10, use_full_dataset=False):
    """
    создать num_loaders случайных выборок (DataLoader) из dataset
    """
    dataloaders = []
    for i in range(num_loaders):
        if use_full_dataset:
            balanced_dataset = dataset
        else:
            balanced_dataset = balancing(dataset, lables_num, dataloader_size)
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
        main_tester.common_time +=  testers[i].common_time
        for j in  range(num_tests):
            main_tester.train_results[j] = (main_tester.train_results[j][0], 
                                            main_tester.train_results[j][1] + testers[i].train_results[j][1], 
                                            main_tester.train_results[j][2] + testers[i].train_results[j][2])
            main_tester.test_results[j] = (main_tester.test_results[j][0],
                                            main_tester.test_results[j][1] + testers[i].test_results[j][1], 
                                            main_tester.test_results[j][2] + testers[i].test_results[j][2])
    main_tester.common_time /= num_testers
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