from main import *

SMD = "SMD"
SMDL2_05 = "SMDL2_05"
SMDL2_01 = "SMDL2_01"
SMDL2_005 = "SMDL2_005"
SMDL2_001 = "SMDL2_001"

def testers_init(method, model, test_dataloader, train_dataloader, device): 
            if method == SMD:
                loss = Loss(loss_fn=nn.CrossEntropyLoss())
            else:
                if method == SMDL2_05:
                    l2_lambda = 0.05
                elif method == SMDL2_01:
                    l2_lambda = 0.01
                elif method == SMDL2_005:
                    l2_lambda = 0.005
                elif method == SMDL2_001:
                    l2_lambda = 0.001
                loss = Loss_L2(loss_fn=nn.CrossEntropyLoss(), l2_lambda=l2_lambda, model_parameters=model.parameters())
            return NetTester(
                model=model,
                device=device,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                optimizer=SMD_qnorm(model.parameters(), lr=0.05, q=3),
                loss=loss,
            )

multi_experiment(       
       setup_datasets=setup_MNIST,
       tester_init=testers_init,
       methods_names=[SMD, SMDL2_05, SMDL2_01, SMDL2_005, SMDL2_001],
       train_sets_num=4,
       labels_num=10,
       train_sizes=[1000, 500, 250, 100],
       train_batches=[200, 100, 50, 20],
       test_batch=256,
       root_folder=os.path.join(EXP_ROOT, 'experiments4'),
       super_model=FCnet(800, 28*28, 10, 2).to(device),
       full_scale=60000,
       full_scale_epochs=40,
       dont_skips=[10, 20, 30, 40],
       test_every=[100, 200, 400, 800])
