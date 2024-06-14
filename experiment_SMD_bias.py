from main import *

SMD = "SMD"
SMD_no_bias = "SMD_no_bias"
SGD = "SGD"

def testers_init(method, model, test_dataloader, train_dataloader, device): 
            lr = 0.01
            model = FCnet(800, 28*28, 10, 2).to(device)
            if method == SGD:
                optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            elif method == SMD:
                   optimizer = SMD_qnorm(model.parameters(), lr=lr, q=3)
            elif method == SMD_no_bias:
                optimizer = SMD_qnorm(model.parameters(), lr=lr, q=3, ignore_bias=True)
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
       methods_names=[SMD_no_bias, SMD, SGD],
       train_sets_num=1,
       labels_num=10,
       train_sizes=[full_scale],
       train_batches=[256],
       test_batch=256,
       root_folder=os.path.join(EXP_ROOT, 'test_no_bias_SMD'),
       super_model=None,
       full_scale=full_scale,
       full_scale_epochs=20,
       dont_skips=[1],
       test_every=[1],
       minimum=True)