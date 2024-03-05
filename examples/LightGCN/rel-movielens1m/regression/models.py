import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import mean_squared_error

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, emb_size):
        super(LightGCN, self).__init__()
        self.user_embedding = nn.Embedding(num_users, emb_size)
        self.item_embedding = nn.Embedding(num_items, emb_size)
        self.gcn_layer = nn.Linear(emb_size, emb_size)

    def forward(self, user_indices, item_indices):
        user_emb = self.user_embedding(user_indices)
        item_emb = self.item_embedding(item_indices)

        # GCN层处理嵌入向量
        user_emb = self.gcn_layer(user_emb)
        item_emb = self.gcn_layer(item_emb)

        return user_emb, item_emb

    def predict(self, item_emb):
        logits = torch.matmul(item_emb, self.item_embedding.weight.t())
        return logits