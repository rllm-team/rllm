# MAE = 0.9377 with 90s on cpu
# simplify LightGCN. original code from: https://github.com/gusye1234/LightGCN-PyTorch is too long to understand so I simplify it.
# independent dataloader in this code

import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import mean_squared_error
from models import LightGCN

# 读取文件
train_df = pd.read_csv('../../../../rllm/datasets/rel-movielens1m/regression/ratings/train.csv')
test_df = pd.read_csv('../../../../rllm/datasets/rel-movielens1m/regression/ratings/test.csv')
val_df = pd.read_csv('../../../../rllm/datasets/rel-movielens1m/regression/ratings/validation.csv')

movie_df = pd.read_csv('../../../../rllm/datasets/rel-movielens1m/regression/movies.csv')
user_df = pd.read_csv('../../../../rllm/datasets/rel-movielens1m/regression/users.csv')

# Create user and movie index mappings
user_idx_map = {user_id: idx for idx, user_id in enumerate(user_df['UserID'].unique())}
movie_idx_map = {movie_id: idx for idx, movie_id in enumerate(movie_df['MovielensID'].unique())}

# 获取训练集中的最大电影索引值和用户索引值
max_movie_idx_train = max(movie_idx_map.values())
max_user_idx_train = max(user_idx_map.values())

# 将用户和电影ID转换为索引
train_df['user_idx'] = train_df['UserID'].apply(lambda x: user_idx_map[x])
train_df['movie_idx'] = train_df['MovieID'].apply(lambda x: movie_idx_map[x])
test_df['user_idx'] = test_df['UserID'].apply(lambda x: user_idx_map.get(x, -1))
test_df['movie_idx'] = test_df['MovieID'].apply(lambda x: movie_idx_map.get(x, -1))

# 构建LightGCN模型
emb_size = 64
num_users = len(user_idx_map)
num_movies = len(movie_idx_map)
model = LightGCN(num_users, num_movies, emb_size)

# 定义损失函数和优化器
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 将数据转换为PyTorch张量
user_indices = torch.tensor(train_df['user_idx'].values, dtype=torch.long)
movie_indices = torch.tensor(train_df['movie_idx'].values, dtype=torch.long)
ratings = torch.tensor(train_df['Rating'].values, dtype=torch.float)

# 训练模型
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

# 在测试集上进行评估（回归任务）
model.eval()
with torch.no_grad():
    model.item_embedding = nn.Embedding(num_movies, emb_size)
    
    user_indices = torch.tensor(test_df['user_idx'].values, dtype=torch.long)
    movie_indices = torch.tensor(test_df['movie_idx'].values, dtype=torch.long)
    ratings = torch.tensor(test_df['Rating'].values, dtype=torch.float)

    # 获取实际的用户和电影嵌入向量
    user_emb, movie_emb = model(user_indices, movie_indices)

    # 预测评分
    predictions = (user_emb * movie_emb).sum(dim=1)

    # 计算平均绝对误差（MAE）
    mae = torch.abs(predictions - ratings).mean()
    print(f'Mean Absolute Error (MAE): {mae.item()}')