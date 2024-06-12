from main import *

SGD2 = "SGD2"
SMD2 = "SMD2"
SGD4 = "SGD4"
SMD4 = "SMD4"

full_scale = 60000
inititial_epochs = 201
repetitions = 2

def tune_init(method, model, test_dataloader, train_dataloader, device):
        lr = 0.01
        model = torch.load(method).to(device)
        if SGD2 in method or SGD4 in method:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        else:
            optimizer = SMD_qnorm(model.parameters(), lr=lr, q=3)
        return NetTester(
            initial_epoch=inititial_epochs,
            model=model,
            device=device,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            optimizer=optimizer,
            loss=Loss(nn.CrossEntropyLoss()),
        )

root = os.path.join(EXP_ROOT, os.path.join('experiments7', 'MNIST60K'))
common_model_name = 'model.pt'
SGD2_models_root = os.path.join(root, 'SGD2')
SGD4_models_root = os.path.join(root, 'SGD4')
models_root = [SGD2_models_root, SGD4_models_root]
models_pathes = []
for r in models_root:
    for i in range(repetitions):
        models_pathes.append(os.path.join(r, os.path.join(f'results_{i+1}', common_model_name)))
print(models_pathes)
multi_experiment(       
       setup_datasets=setup_MNIST, # setup_MNIST
       tester_init=tune_init,
       methods_names=models_pathes,
       train_sets_num=1,
       labels_num=10,
       train_sizes=[full_scale],
       train_batches=[256],
       test_batch=256,
       root_folder=os.path.join(EXP_ROOT, 'test_experiments7_tune'),
       super_model=None,
       full_scale=full_scale,
       full_scale_epochs=2,
       dont_skips=[20],
       test_every=[10],
       minimum=True)