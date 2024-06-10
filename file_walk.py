import os
import re

def find_files(directory, filename):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == filename:
                file_paths.append(os.path.join(root, file))
    return file_paths

def read_last_line(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        if lines:
            return lines[-1]
        else:
            return "Файл пустой"

def remove_digits_and_underscore(input_string):
    result = re.sub(r'[\d_]', '', input_string)
    return result
# semilicon - заменить точку на запятую
def print_res(dict, semilicon=False):
    for key, value in dict.items():
        SGD =  value[0].value
        SMD = value[1].value
        if value[0].method == 'SMD':
            SGD, SMD = SMD, SGD
        if semilicon:
            SGD = str(SGD).replace('.', ',')
            SMD = str(SMD).replace('.', ',')
        print(key, SGD, SMD)

class Result:
    def __init__(self, method, measurement, value):
        self.method = method
        self.measurement = measurement
        self.value = value

class Pair:
    def __init__(self, res1, res2):
        self.result1 = res1
        self.result2 = res2

#directory = "D:/github/mirror-descent/experiments/for_report/experiments1" #PC
directory = "D:/WorkProjects/github/mirror-descent/experiments/for_report/experiments1" #Laptop
#filename = "test_results.txt" 
filename = "train_results.txt"

result = find_files(directory, filename)
accuracy_results = {}
loss_results = {}
for file_path in result:
    root_split = file_path.split('/')
    inside_split = root_split[len(root_split)-1].split('\\')
    if inside_split[1] == 'SGDL2':
        continue
    sample_size = inside_split[1]
    method = remove_digits_and_underscore(inside_split[2])
    last_line = read_last_line(file_path).split(' ')
    accuracy = last_line[1]
    loss = last_line[2][:-1]
    res_acc = Result(method=method, measurement='accuracy', value=accuracy)
    res_loss = Result(method=method, measurement='loss', value=loss)
    if sample_size in accuracy_results:
        accuracy_results[sample_size].append(res_acc)
        loss_results[sample_size].append(res_loss)
    else:
        accuracy_results[sample_size] = [res_acc]   
        loss_results[sample_size] = [res_loss]  
    #print(inside_split[1], remove_digits_and_underscore(inside_split[2]), read_last_line(file_path))

sorted_acc = dict(sorted(accuracy_results.items(), key=lambda x: int(x[0])))
sorted_loss = dict(sorted(loss_results.items(), key=lambda x: int(x[0])))
semicolon = True
print_res(sorted_acc, semicolon)
print_res(sorted_loss, semicolon)