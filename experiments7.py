from main import *

SGD2 = "SGD2"
SMD2 = "SMD2"
SGD4 = "SGD4"
SMD4 = "SMD4"

def testers_init(method, model, test_dataloader, train_dataloader, device): 
            lr = 0.01
            if method == SGD2:
                model = FCnet(800, 28*28, 10, 2).to(device)
                optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            elif method == SGD4:
                   model = FCnet(800, 28*28, 10, 4).to(device)
                   optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            elif method == SMD2:
                model = FCnet(800, 28*28, 10, 2).to(device)
                optimizer = SMD_qnorm(model.parameters(), lr=lr, q=3)
            elif method == SMD4:
                model = FCnet(800, 28*28, 10, 4).to(device)
                optimizer = SMD_qnorm(model.parameters(), lr=lr, q=3)
            return NetTester(
                model=model,
                device=device,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                optimizer=optimizer,
                loss=Loss(nn.CrossEntropyLoss()),
            )

full_scale = 60000
multi_experiment(       
       setup_datasets=setup_MNIST, # setup_MNIST
       tester_init=testers_init,
       methods_names=[SGD2, SMD2, SGD4, SMD4],
       train_sets_num=4,
       labels_num=10,
       train_sizes=[1000],
       train_batches=[25],
       test_batch=256,
       root_folder=os.path.join(EXP_ROOT, 'experiments7_MNIST1000'),
       super_model=None,
       full_scale=full_scale,
       full_scale_epochs=10,
       dont_skips=[40],
       test_every=[20])