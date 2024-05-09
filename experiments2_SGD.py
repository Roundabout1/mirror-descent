"""
SGD-обучение и сохранение моделей вместе с исходными обучающими данными
"""
from experiment2_main import *

# загрузка начальных данных
train_dataset, test_dataset = setup()
test_dataloader = DataLoader(test_dataset, batch_size=TEST_BATCH)
train_dataloaders = make_dataloaders(TRAIN_SAMPLE_NUM, train_dataset, TRAIN_DATA_SIZE, TRAIN_BATCH, 10)

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
      show_progress=False
    )
    SGD_testers.append(cur_tester)

run_tests(SGD_testers, SGD_EPOCHS*(60000//TRAIN_DATA_SIZE), 50, 1000)

for tester in SGD_testers:
    tester.save_results(SGD_MODELS, 'SGD')

main_tester = concat_results(SGD_testers)
main_tester.save_results(SGD_MODELS, 'average_results', True)