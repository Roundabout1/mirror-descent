import torch.nn as nn

# neurons_num - количество нейронов в каждом слое
# input_size - размер входных данных
# output_size - размер выходных данных
# inner_layers_num - количество внутренних (помимо входного и выходного) слоёв
class FCnet(nn.Module):
    def __init__(self, neurons_num, input_size, output_size, inner_layers_num):
        super(FCnet, self).__init__()
        self.neurons_num = neurons_num
        self.input_size = input_size
        self.output_size = output_size
        # внутренние слои + входной + выходной
        self.layers_num = inner_layers_num + 2
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, neurons_num))
        for i in range(inner_layers_num):
            self.layers.append(nn.Linear(neurons_num, neurons_num))
        self.layers.append(nn.Linear(neurons_num, output_size))
        self.activation_function = nn.functional.relu

    def forward(self, x):
        x = x.view(-1, self.input_size)
        for i in range(self.layers_num-1):
            x = self.activation_function(self.layers[i](x))
        x = self.layers[self.layers_num-1](x)
        return x
    def info(self):
        layers = f'layers: {self.layers_num}\n'
        neurons = f'neurons_num: {self.neurons_num}\n'
        img = f'img_size: {self.input_size}\n'
        active_fun = f'activation function: {self.activation_function.__name__}\n'
        return layers + neurons + img + active_fun