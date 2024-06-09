"""
тестирование loss с и без учёта сдвига (bias)
"""
from main import *

SGD_bias = "SGD_bias"
SGD_no_bias = "SGD_no_bias"
SGD_no_L2 = "SGD_no_l2"
SGD_default_L2 = "SGD_default_L2"

def testers_init(method, model, test_dataloader, train_dataloader, device): 
            lr = 0.05
            model = FCnet(800, 28*28, 10, 4).to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            if method == SGD_bias:
                loss = Loss_L2(model=model, loss_fn=nn.CrossEntropyLoss(), l2_lambda=0.001, ignore_bias=False)
            if method == SGD_no_bias:
                loss = Loss_L2(model=model, loss_fn=nn.CrossEntropyLoss(), l2_lambda=0.001, ignore_bias=True)
            if method == SGD_no_L2:
                loss = Loss(loss_fn=nn.CrossEntropyLoss())
            if method == SGD_default_L2:
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.001)
                loss = Loss(loss_fn=nn.CrossEntropyLoss())
            return NetTester(
                model=model,
                device=device,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                optimizer=optimizer,
                loss=loss)

full_scale = 60000
multi_experiment(       
       setup_datasets=setup_MNIST,
       tester_init=testers_init,
       methods_names=[SGD_bias, SGD_no_bias, SGD_default_L2, SGD_no_L2],
       train_sets_num=1,
       labels_num=5,
       train_sizes=[full_scale],
       train_batches=[256],
       test_batch=256,
       root_folder=os.path.join(EXP_ROOT, 'SGD_bias_nobias_defaultL2_noL2'),
       super_model=None,
       full_scale=full_scale,
       full_scale_epochs=10,
       dont_skips=[20],
       test_every=[10])