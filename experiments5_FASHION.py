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
       train_sizes=[30000, 20000, 10000, 5000],
       train_batches=[256, 256, 256, 256],
       test_batch=256,
       root_folder=os.path.join(EXP_ROOT, 'experiments5_fashion_medium'),
       super_model=FCnet(800, 28*28, 10, 4).to(device),
       full_scale=full_scale,
       full_scale_epochs=100,
       dont_skips=[2, 3, 6, 12],
       test_every=[2, 3, 6, 12])