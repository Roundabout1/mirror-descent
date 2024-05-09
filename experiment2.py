"""
обучение переобученных моделей методом зеркального спуска

сначала модели обучаются посредством SGD, затем SMD

---В ПРОЦЕССЕ НАПИСАНИЯ---
"""
from torch.utils.data import DataLoader
from FCnet import FCnet
from loss import Loss_L2, Loss
from main import concat_results, make_dataloaders, make_models, run_tests, setup, device, EXP_ROOT 
from tester import NetTester
from SMD_opt import SMD_qnorm
import torch.nn as nn
import torch.optim as optim
import os
"""
корневая папка для текущего эксперимента
"""
CUR_ROOT = os.path.join(EXP_ROOT, "experiments2")
"""
папка к исходным обучающим данным
"""
TRAIN_DATA = os.path.join(CUR_ROOT, 'train_data')
"""
папка с переобученными моделями (натренированные через SGD)
"""
SGD_MODELS = os.path.join(CUR_ROOT, 'sgd_models')
"""
папка с моделями, обученными через SMD
"""
SMD_MODELS = os.path.join(CUR_ROOT, 'smd_models')

# загрузка начальных данных
train_dataset, test_dataset = setup()
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=True)
train_dataloaders = make_dataloaders(4, train_dataset, 1000, 100, 10)

# SGD
SUPER_MODEL = FCnet(800, 28*28, 10, 2).to(device)
models = make_models(len(train_dataloaders), SUPER_MODEL)
# создаём тестеры
num_testers = len(models)
SGD_testers = []
for i in range(num_testers):
    cur_tester = NetTester(
      model=models[i],
      device=device,
      train_dataloader=train_dataloaders[i],
      test_dataloader=test_dataloader,
      optimizer=optim.SGD(models[i].parameters(), lr=0.05),
      loss=Loss(loss_fn=nn.CrossEntropyLoss()),
      show_progress=True
    )
    SGD_testers.append(cur_tester)

run_tests(SGD_testers, 20, 5, 2)

for tester in SGD_testers:
    tester.save_results(SGD_MODELS, 'SGD')

main_tester = concat_results(SGD_testers)
main_tester.save_results(SGD_MODELS, 'average_results', True)

# SMD
SMD_testers = []
for i in range(num_testers):
    cur_tester = NetTester(
      model=models[i],
      device=device,
      train_dataloader=train_dataloaders[i],
      test_dataloader=test_dataloader,
      optimizer=optim.SGD(models[i].parameters(), lr=0.05),
      loss=Loss(loss_fn=nn.CrossEntropyLoss()),
      show_progress=True,
      initial_epoch=SGD_testers[i].actual_epochs+1
    )
    SMD_testers.append(cur_tester)

run_tests(SMD_testers, 20, 5, 2)

for tester in SMD_testers:
    tester.save_results(SMD_MODELS, 'SMD')

main_tester = concat_results(SMD_testers)
main_tester.save_results(SMD_MODELS, 'average_results', True)