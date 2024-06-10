from main import *

SGD = "SGD"
SMD = "SMD"

def testers_init(method, model, test_dataloader, train_dataloader, device): 
            lr = 0.01
            if method == SGD:
                optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            else:
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
       setup_datasets=setup_FASHION,
       tester_init=testers_init,
       methods_names=[SMD, SGD],
       train_sets_num=4,
       labels_num=10,
       train_sizes=[500, 250, 100],
       train_batches=[50, 50, 50],
       test_batch=256,
       root_folder=os.path.join(EXP_ROOT, 'experiments5_fashion_small'),
       super_model=FCnet(800, 28*28, 10, 4).to(device),
       full_scale=full_scale,
       full_scale_epochs=8,
       dont_skips=[1, 1, 1],
       test_every=[20, 40, 80])