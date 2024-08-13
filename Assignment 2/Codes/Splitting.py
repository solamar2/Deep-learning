import random

def read_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines[1:]]

def split_data(data):
    names_3 = {}
    names_4 = {}

    for line in data:
        parts = line.split()
        if len(parts) == 3:
            name = parts[0]
            if name in names_3:
                names_3[name].append(line)
            else:
                names_3[name] = [line]
        elif len(parts) == 4:
            name = (parts[0], parts[2])
            if name in names_4:
                names_4[name].append(line)
            else:
                names_4[name] = [line]
    
    # Shuffle and split names_3
    all_names_3 = list(names_3.keys())
    random.shuffle(all_names_3)
    split_point_3 = int(0.8 * len(all_names_3))
    train_names_3 = set(all_names_3[:split_point_3])
    validation_names_3 = set(all_names_3[split_point_3:])
    
    # Shuffle and split names_4
    all_names_4 = list(names_4.keys())
    random.shuffle(all_names_4)
    split_point_4 = int(0.8 * len(all_names_4))
    train_names_4 = set(all_names_4[:split_point_4])
    validation_names_4 = set(all_names_4[split_point_4:])
    
    train_data = []
    validation_data = []
    
    for name in train_names_3:
        train_data.extend(names_3[name])
        
    for name in validation_names_3:
        validation_data.extend(names_3[name])
        
    for name in train_names_4:
        train_data.extend(names_4[name])
        
    for name in validation_names_4:
        validation_data.extend(names_4[name])
    
    return train_data, validation_data

def write_data(data, file_path):
    with open(file_path, 'w') as file:
        for line in data:
            file.write(line + '\n')

# Paths to input and output files
input_file_path = 'C:/Users/idogu/deep/work 2/pairsDevTrain.txt'
train_file_path = 'C:/Users/idogu/deep/work 2/trainset.txt'
validation_file_path ='C:/Users/idogu/deep/work 2/validationset.txt'

# Read, split, and write the data
data = read_data('C:/Users/idogu/deep/work 2/pairsDevTrain.txt')
train_data, validation_data = split_data(data)
write_data(train_data, train_file_path)
write_data(validation_data, validation_file_path)



