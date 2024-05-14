"""
параметры для эксперимента, суть которого заключается в следующем:

обучение переобученных моделей методом зеркального спуска

сначала модели обучаются посредством SGD, затем SMD
"""
from torch.utils.data import DataLoader
from FCnet import FCnet
from loss import Loss_L2, Loss
from main import concat_results, make_dataloaders, make_models, run_tests, setup_MNIST, device, EXP_ROOT 
from tester import NetTester
from SMD_opt import SMD_qnorm
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torchvision import datasets, transforms

SGD_EPOCHS = 40
"""
кол-во эпох, потраченных на обучение
"""

TRAIN_SAMPLE_NUM = 4
"""
кол-во обучающих выборок
"""

TRAIN_DATA_SIZE = 100
"""
размер обучающих выборок
"""

TRAIN_BATCH = 10
"""
рамер пакета (batch) у tain dataloader
"""

TEST_BATCH = 256

CUR_ROOT = os.path.join(EXP_ROOT, "experiments2")
"""
корневая папка для текущего эксперимента
"""

SGD_MODELS = os.path.join(CUR_ROOT, 'SGD_models')
"""
папка с переобученными моделями (натренированные через SGD)
"""

SUPER_MODEL = FCnet(800, 28*28, 10, 2).to(device)
"""
шаблон модели для копирования
"""