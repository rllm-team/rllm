# This file creates knowledge graph needed to run regression task, the output files are:
# train.txt, test.txt, val.txt: User/Movie interactions of train/test/val set
# test_rating.txt, val_rating.txt: User-rating-movie triplets of test/val set
import sys
sys.path.append("../../../../rllm/dataloader")
sys.path.append("../../../kgat")

import pandas as pd
import torch
# from utils import load_data

from load_data import load_data
from utils.sample import adj_matrix_to_list, value_matrix_to_list

# Load data
data, adj, features, labels, idx_train, idx_val, idx_test = \
    load_data('movielens-regression')
# labels = labels.argmax(dim=-1)

head_tail = adj.indices().detach().clone()
movie_shift = head_tail[1, :].min()
head_tail[1, :] = head_tail[1, :] - movie_shift
head_tail = head_tail.numpy()
relation = labels.int().numpy()
print("writing knowledge_graph")
df = pd.DataFrame({
    'h': head_tail[0, :],  # First row of heads_tails tensor
    'r': relation,      # Relations
    't': head_tail[1, :]   # Second row of heads_tails tensor
})
df.to_csv('kg_final.txt', sep=' ', index=False, header=False)

# train_interaction = adj.indices()[:, idx_train]
# test_interaction = adj.indices()[:, idx_test]
# val_interaction = adj.indices()[:, idx_val]


# node_index = torch.arange(0, adj.shape[0])
node_index_user = torch.arange(0, adj.indices()[0, -1]+1)
# node_index_movie = torch.arange(adj.indices()[1, 0], adj.indices()[1, -1]+1)
node_index_train_user = \
    torch.unique(adj.indices()[:, idx_train][0], return_counts=False)
node_index_train_movie = \
    torch.unique(adj.indices()[:, idx_train][1], return_counts=False)
node_index_test_user = \
    torch.unique(adj.indices()[:, idx_test][0], return_counts=False)
node_index_test_movie = \
    torch.unique(adj.indices()[:, idx_test][1], return_counts=False)
node_index_val_user = \
    torch.unique(adj.indices()[:, idx_val][0], return_counts=False)
node_index_val_movie = \
    torch.unique(adj.indices()[:, idx_val][1], return_counts=False)

train_mat = torch.sparse_coo_tensor(
    adj.indices()[:, idx_train],
    labels[idx_train],
    size=adj.shape,
    requires_grad=False).float()
test_mat = torch.sparse_coo_tensor(
    adj.indices()[:, idx_test],
    labels[idx_test],
    size=adj.shape,
    requires_grad=False).float()
val_mat = torch.sparse_coo_tensor(
    adj.indices()[:, idx_val],
    labels[idx_val],
    size=adj.shape,
    requires_grad=False).float()

train_interaction = adj_matrix_to_list(train_mat)
test_interaction = adj_matrix_to_list(test_mat)
val_interaction = adj_matrix_to_list(val_mat)

test_rating = value_matrix_to_list(test_mat)
val_rating = value_matrix_to_list(val_mat)


def shift_movie_idx(value):
    return str((value - movie_shift).item())


print("writing train_interaction")

# Open a text file to write
with open('train.txt', 'w') as file:
    for key, values in train_interaction.items():
        if key >= movie_shift:
            continue
        # Create a string
        # where the key is followed by all the values separated by spaces
        line = f'{key} ' + ' '.join(map(shift_movie_idx, values)) + '\n'
        # Write the formatted string to the file
        file.write(line)
print("writing test_interaction")


# Open a text file to write
with open('test.txt', 'w') as file:
    for key, values in test_interaction.items():
        if key >= movie_shift:
            continue
        # Create a string
        # where the key is followed by all the values separated by spaces
        line = f'{key} ' + ' '.join(map(shift_movie_idx, values)) + '\n'
        # Write the formatted string to the file
        file.write(line)
print("writing test_rating")
# Open a text file to write
with open('test_rating.txt', 'w') as file:
    for key, values in test_rating.items():
        if key >= movie_shift:
            continue
        # Create a string where the key is followed by
        # all the values separated by spaces
        line = f'{key} ' + ' '.join(map(str, values)) + '\n'
        # Write the formatted string to the file
        file.write(line)
print("writing val_interaction")
# Open a text file to write
with open('val.txt', 'w') as file:
    for key, values in val_interaction.items():
        if key >= movie_shift:
            continue
        # Create a string
        # where the key is followed by all the values separated by spaces
        line = f'{key} ' + ' '.join(map(shift_movie_idx, values)) + '\n'
        # Write the formatted string to the file
        file.write(line)
print("writing val_rating")
# Open a text file to write
with open('val_rating.txt', 'w') as file:
    for key, values in val_rating.items():
        if key >= movie_shift:
            continue
        # Create a string where the key is followed by
        # all the values separated by spaces
        line = f'{key} ' + ' '.join(map(str, values)) + '\n'
        # Write the formatted string to the file
        file.write(line)
