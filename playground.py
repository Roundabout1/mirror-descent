from torch.utils.data import DataLoader
from FCnet import FCnet
from loss import Loss_L2, Loss
from main import concat_results, make_dataloaders, make_models, run_tests, setup_MNIST, device
from tester import NetTester
from SMD_opt import SMD_qnorm
import torch.nn as nn
import torch.optim as optim
train_dataset, test_dataset = setup_MNIST()
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=True)
train_dataloaders = make_dataloaders(2, train_dataset, 1000, 100, 10)
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
      optimizer=SMD_qnorm(models[i].parameters(), lr=0.05, q=3),
      loss=Loss_L2(loss_fn=nn.CrossEntropyLoss(), model_parameters=models[i].parameters(), l2_lambda=0.01),
      show_progress=True
    )
    testers.append(cur_tester)
run_tests(testers, 10, 5, 2)
for i in testers:
    print(i.common_time)
main_tester = concat_results(testers)
main_tester.save_results("test_folder", 'test', minimum=False)