# modified LightGCN for classification task in Cora
# Paper: original code from: https://github.com/gusye1234/LightGCN-PyTorch is too long to understand so I simplify it
# MAE = 0.9377
# Runtime: 90s on cpu
# Cost: N/A
# Description: In order to simplify lightGCN, we only reserve the basic idea of the lightGCN architecture:
# Description: only single-layer linear transformation. 
# Description:In this model, we only retain the final result output instead of using multiple layers of output averaging as in the standard LightGCN.

import torch
import torch.nn as nn
import pandas as pd

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, emb_size):
        super(LightGCN, self).__init__()
        self.user_embedding = nn.Embedding(num_users, emb_size)
        self.item_embedding = nn.Embedding(num_items, emb_size)
        self.gcn_layer = nn.Linear(emb_size, emb_size)

    def forward(self, user_indices, item_indices):
        user_emb = self.user_embedding(user_indices)
        item_emb = self.item_embedding(item_indices)

        # get embedding
        user_emb = self.gcn_layer(user_emb)
        item_emb = self.gcn_layer(item_emb)

        return user_emb, item_emb

    def predict(self, item_emb):
        logits = torch.matmul(item_emb, self.item_embedding.weight.t())
        return logits

# 读取文件
train_df = pd.read_csv('../../../rllm/datasets/rel-movielens1m/regression/ratings/train.csv')
test_df = pd.read_csv('../../../rllm/datasets/rel-movielens1m/regression/ratings/test.csv')
val_df = pd.read_csv('../../../rllm/datasets/rel-movielens1m/regression/ratings/validation.csv')

movie_df = pd.read_csv('../../../rllm/datasets/rel-movielens1m/regression/movies.csv')
user_df = pd.read_csv('../../../rllm/datasets/rel-movielens1m/regression/users.csv')

# Create user and movie index mappings
user_idx_map = {user_id: idx for idx, user_id in enumerate(user_df['UserID'].unique())}
movie_idx_map = {movie_id: idx for idx, movie_id in enumerate(movie_df['MovielensID'].unique())}


# map user_id and movie_id
train_df['user_idx'] = train_df['UserID'].apply(lambda x: user_idx_map[x])
train_df['movie_idx'] = train_df['MovieID'].apply(lambda x: movie_idx_map[x])
test_df['user_idx'] = test_df['UserID'].apply(lambda x: user_idx_map.get(x, -1))
test_df['movie_idx'] = test_df['MovieID'].apply(lambda x: movie_idx_map.get(x, -1))

# create the model
emb_size = 64
num_users = len(user_idx_map)
num_movies = len(movie_idx_map)
model = LightGCN(num_users, num_movies, emb_size)

# create the loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# change into tensor
user_indices = torch.tensor(train_df['user_idx'].values, dtype=torch.long)
movie_indices = torch.tensor(train_df['movie_idx'].values, dtype=torch.long)
ratings = torch.tensor(train_df['Rating'].values, dtype=torch.float)

print(user_indices.shape, movie_indices.shape, ratings.shape)

# train process begin
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    user_emb, movie_emb = model(user_indices, movie_indices)
    predictions = (user_emb * movie_emb).sum(dim=1)
    loss = loss_fn(predictions, ratings)
    loss.backward()
    optimizer.step()
    if epoch % 49 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# test process begin
model.eval()
with torch.no_grad():
    model.item_embedding = nn.Embedding(num_movies, emb_size)
    
    user_indices = torch.tensor(test_df['user_idx'].values, dtype=torch.long)
    movie_indices = torch.tensor(test_df['movie_idx'].values, dtype=torch.long)
    ratings = torch.tensor(test_df['Rating'].values, dtype=torch.float)

    # get embedding
    user_emb, movie_emb = model(user_indices, movie_indices)
    
    # get results
    predictions = (user_emb * movie_emb).sum(dim=1)

    # get MAE
    mae = torch.abs(predictions - ratings).mean()
    print(f'Mean Absolute Error (MAE): {mae.item()}')