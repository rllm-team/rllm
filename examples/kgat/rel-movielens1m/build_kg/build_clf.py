import sys
import os
sys.path.append("../../../../rllm/dataloader")
sys.path.append("../../../kgat")

# import pandas as pd
# import torch
# from utils import load_data

from load_data import load_data
# from utils.sample import adj_matrix_to_list, value_matrix_to_list

# Load data
data, adj, features, labels, idx_train, idx_val, idx_test = \
    load_data('movielens-classification')
# labels = labels.argmax(dim=-1)

train_cat_dict = {}
test_cat_dict = {}
val_cat_dict = {}

for i, label in labels.nonzero():
    if i in idx_train:
        if i.item() not in train_cat_dict.keys():
            train_cat_dict[i.item()] = [label.item()]
        else:
            train_cat_dict[i.item()].append(label.item())
    elif i in idx_test:
        if i.item() not in test_cat_dict.keys():
            test_cat_dict[i.item()] = [label.item()]
        else:
            test_cat_dict[i.item()].append(label.item())
    elif i in idx_val:
        if i.item() not in val_cat_dict.keys():
            val_cat_dict[i.item()] = [label.item()]
        else:
            val_cat_dict[i.item()].append(label.item())
print("writing train category")
# Open a text file to write
os.makedirs("../datasets/rel-movielens/", exist_ok=True)
with open('../datasets/rel-movielens/train_category.txt', 'w') as file:
    for key, values in train_cat_dict.items():
        # Create a string
        # where the key is followed by all the values separated by spaces
        line = f'{key} ' + ' '.join(map(str, values)) + '\n'
        # Write the formatted string to the file
        file.write(line)
print("writing test category")
# Open a text file to write
with open('../datasets/rel-movielens/test_category.txt', 'w') as file:
    for key, values in test_cat_dict.items():
        # Create a string
        # where the key is followed by all the values separated by spaces
        line = f'{key} ' + ' '.join(map(str, values)) + '\n'
        # Write the formatted string to the file
        file.write(line)
print("writing val category")
# Open a text file to write
with open('../datasets/rel-movielens/val_category.txt', 'w') as file:
    for key, values in val_cat_dict.items():
        # Create a string where the key is followed by
        # all the values separated by spaces
        line = f'{key} ' + ' '.join(map(str, values)) + '\n'
        # Write the formatted string to the file
        file.write(line)
