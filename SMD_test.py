"""
Тест работы SMD
"""
from main import *

SMD = "SMD"

def testers_init(method, model, test_dataloader, train_dataloader, device): 
            model = FCnet(800, 28*28, 10, 4).to(device)
            return NetTester(
                model=model,
                device=device,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                optimizer= SMD_qnorm(model.parameters(), lr=0.05, q=3),
                loss=Loss(nn.CrossEntropyLoss()),
            )

full_scale = 60000
multi_experiment(       
       setup_datasets=setup_MNIST, # setup_MNIST
       tester_init=testers_init,
       methods_names=[SMD],
       train_sets_num=4,
       labels_num=10,
       train_sizes=[60000],
       train_batches=[256],
       test_batch=256,
       root_folder=os.path.join(EXP_ROOT, 'SMD_MNIST_FULL_TEST'),
       super_model=None,
       full_scale=full_scale,
       full_scale_epochs=10,
       dont_skips=[40],
       test_every=[20])