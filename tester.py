import os
import time
import torch
import numpy as np

class TrainLog:
    def __init__(self, epochs, first_epoch, dont_skip, test_every):
        self.epochs = epochs
        self.first_epoch = first_epoch
        self.dont_skip = dont_skip
        self.test_every = test_every

# тестировщик нейронной сети
class NetTester:
    # initial_epoch - эпоха с которой следует начинать нумерацию
    def __init__(self, model, device, train_dataloader, test_dataloader, optimizer, loss_fn, initial_epoch=1):
        self.model = model
        self.device = device
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.initial_epoch = initial_epoch
        self.actual_epochs = 0
        self.dont_skip = 0
        self.test_every = 0
        self.train_results = []
        self.test_results = []
        self.training_log = []
    # обучение в течение одной эпохи
    def train_step(self, dataloader, show_progress=False):
        size = len(dataloader.dataset)
        self.model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)

            # Ошибка предсказания
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # вывод текущего прогресса, для того, чтобы убедиться, что обучение идёт
            if show_progress and batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(self, dataloader):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        return 100*correct, test_loss

    # обучение модели и получение результатов обучения на тестовой и обучающих выборках
    # dont_skip - до какой эпохи не пропускать тесты (значение меньше нуля будет означать не пропускать тесты)
    # test_every - тестировать каждую test_every эпоху, все остальноё - пропустить
    def train(self, epochs, dont_skip=-1, test_every=1):
        if dont_skip < 0:
            dont_skip = epochs
        self.training_log.append(TrainLog(epochs=epochs, dont_skip=dont_skip, test_every=test_every, first_epoch=self.actual_epochs+1))
        self.dont_skip = dont_skip
        self.test_every = test_every
        for e in range(epochs):
            print(f"Epoch {self.actual_epochs+1}\n-------------------------------")
            self.train_step(self.train_dataloader)
            if e < dont_skip or e % test_every == 0 or e == epochs-1:
                train_accuracy, train_loss = self.test(self.train_dataloader)
                self.train_results.append((self.initial_epoch+self.actual_epochs, train_accuracy, train_loss))
                test_accuracy, test_loss = self.test(self.test_dataloader)
                self.test_results.append((self.initial_epoch+self.actual_epochs, test_accuracy, test_loss))
            self.actual_epochs += 1 
        print("Done!")
        return self.train_results, self.test_results

    # запись результатов тестирования 
    def save_results(self, output_root, folder_name='treck'):
        # генерация уникального имени

        cur_time = (str(time.time())).replace('.', '_')
        folder_path_base = os.path.join(output_root, 'treck_' + cur_time)
        # проверка на то, что это имя не существует
        cnt = 0 
        folder_path = folder_path_base
        while os.path.isdir(folder_path):
            folder_path = folder_path_base + '_' + str(cnt) 
            cnt += 1 
        os.makedirs(folder_path, exist_ok=True)

        np.savetxt(os.path.join(folder_path, 'train_results.txt'), np.array(self.train_results), fmt='%d %.2f %.8f')
        np.savetxt(os.path.join(folder_path, 'test_results.txt'), np.array(self.test_results), fmt='%d %.2f %.8f')
        with open(os.path.join(folder_path, 'info.txt'), 'w+') as f:
            f.write(self.model.info())
            f.write(f'Epochs: {self.actual_epochs}\n')
            f.write(f'unskippable epochs: {self.dont_skip}\n')
            f.write(f'test every {self.test_every} epochs\n')
            f.write(f'loss function: {self.loss_fn}\n')
            f.write(f'optimizer: {self.optimizer}\n')
            f.write(f'train batch size: {self.train_dataloader.batch_size}\n')
            f.write(f'test batch size: {self.test_dataloader.batch_size}\n')
            f.write(f'train data size: {len(self.train_dataloader.dataset)}\n')
        # сохранение модели
        torch.save(self.model, os.path.join(folder_path, 'model.pt'))
        # сохранение тренировочных данных
        torch.save(self.train_dataloader.dataset, os.path.join(folder_path, 'train.pt'))