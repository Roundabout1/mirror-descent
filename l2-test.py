from main import *

# SGD = "SGD"
SGDL2 = "SGDL2"
SGDL2_CUSTOM = "SGDL2_CUSTOM"


def testers_init(method, model, test_dataloader, train_dataloader, device):           
            if method == SGDL2:
                loss = Loss(loss_fn=nn.CrossEntropyLoss())
                weight_decay = 0.01
            else:
                loss = Loss_L2(loss_fn=nn.CrossEntropyLoss(), l2_lambda=0.01, model=model)
                weight_decay = 0
            return NetTester(
                model=model,
                device=device,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                optimizer=optim.SGD(model.parameters(), lr=0.05, weight_decay=weight_decay),
                loss= loss,
            )

multi_experiment(       
       setup_datasets=setup_MNIST,
       tester_init=testers_init,
       methods_names=[SGDL2_CUSTOM, SGDL2],
       train_sets_num=3,
       labels_num=10,
       train_sizes=[100],
       train_batches=[20],
       test_batch=256,
       root_folder=os.path.join(EXP_ROOT, 'L2-tests'),
       super_model=FCnet(800, 28*28, 10, 2).to(device),
       full_scale=60000,
       full_scale_epochs=20,
       dont_skips=[25],
       test_every=[800])
