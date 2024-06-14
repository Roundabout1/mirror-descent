import torch
import numpy as np
import random
# Разбиение обучающих данных
# labels_num -  количество меток (от 0 до 9)
# train_len - длина того обучающего множества, которое мы хотим использовать для обучения, оно должно делиться на количество меток
def balancing(full_dataset, lables_num, desirable_len):
    # Длина всего обучающего множества и того обучающего множества, которое мы хотим использовать для обучения
    full_train_len = len(full_dataset)
    # Количество данных с одной меткой
    label_group_num = int(desirable_len/lables_num)

    # Создаём группы для хранения индексов каждой метки в обучающем наборе данных
    label_groups_index = [[] for _ in range(lables_num)]
    for i in range(full_train_len):
        label = full_dataset[i][1]
        label_groups_index[label].append(i)

    # Обрезаем группы, оставляя случайные, неповторяющиеся элементы в каждой и объединяем их всех в один набор индексов
    all_index = np.array([], dtype=int)
    for i in range(lables_num):
        all_index = np.append(all_index, random.sample(label_groups_index[i], label_group_num))
    np.random.shuffle(all_index)

    # Формируем обучающий набор данных
    train_dataset = torch.utils.data.Subset(full_dataset, all_index)
    return train_dataset