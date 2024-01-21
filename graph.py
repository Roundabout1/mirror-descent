import matplotlib.pyplot as plt

# Чтение данных из файлов
with open('output/test.txt', 'r') as f:
    test_data = f.readlines()
with open('output/train.txt', 'r') as f:
    train_data = f.readlines()

# Извлечение точности и функции потерь из данных
test_accuracy = [float(line.split()[0]) for line in test_data]
test_loss = [float(line.split()[1]) for line in test_data]
train_accuracy = [float(line.split()[0]) for line in train_data]
train_loss = [float(line.split()[1]) for line in train_data]

# Построение графиков
fig, axs = plt.subplots(2)
fig.suptitle('Точность и функция потерь за каждую эпоху обучения нейронной сети')
axs[0].plot(test_accuracy, label='Тестовая выборка')
axs[0].plot(train_accuracy, label='Обучающая выборка')
axs[0].set_ylabel('Точность')
axs[0].legend()
axs[1].plot(test_loss, label='Тестовая выборка')
axs[1].plot(train_loss, label='Обучающая выборка')
axs[1].set_xlabel('Эпоха')
axs[1].set_ylabel('Функция потерь')
axs[1].legend()
plt.show()
