from experiments2_main import *

SMD_MODELS = os.path.join(EXP_ROOT, 'SMD_models')

models = []
train_dataloaders = []
for i in range(TRAIN_SAMPLE_NUM):
    path = os.path.join(SGD_MODELS, f'SGD_{i+1}')

    # загрузка моделей
    model = torch.load(os.path.join(path, 'model.pt'))
    #model.train()
    model.to(device)
    models.append(model)

    # загрузка датасетов
    train_dataset = torch.load(os.path.join(path, 'train.pt'))
    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH, shuffle=False)
    train_dataloaders.append(train_dataloader)

test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
test_dataloader = DataLoader(test_dataset, batch_size=TEST_BATCH)

num_testers = len(models)
SMD_testers = []
for i in range(num_testers):
    cur_tester = NetTester(
      model=models[i],
      device=device,
      train_dataloader=train_dataloaders[i],
      test_dataloader=test_dataloader,
      optimizer=SMD_qnorm(models[i].parameters(), lr=0.005, q=3),
      loss=Loss(loss_fn=nn.CrossEntropyLoss()),
      show_progress=False,
      initial_epoch=SGD_EPOCHS+1
    )
    SMD_testers.append(cur_tester)

run_tests(SMD_testers, 1000, 25, 25)

for tester in SMD_testers:
    tester.save_results(SMD_MODELS, 'SMD')

main_tester = concat_results(SMD_testers)
main_tester.save_results(SMD_MODELS, 'average_results', True)