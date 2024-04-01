# Deepfm for regression task 
# Paper: DeepFM: A Factorization-Machine based Neural Network for CTR Prediction
# arxiv : https://arxiv.org/abs/1703.04247
# MAE: 0.899
# Runtime: 9.10s on CPU with 10 cores(Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz)
# Cost: N/A
# Description: Paper Reproduction.

import numpy as np
import pandas as pd
from torch import nn, optim
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import sys
sys.path.append("../../src")
from inputs import SparseFeat, VarLenSparseFeat, build_input_array
from models import DeepFM
from snippets import sequence_padding, seed_everything
from sklearn.metrics import mean_squared_error
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed_everything(37) 

def get_data():
    def split(x):
        key_ans = x.split('|')
        for key in key_ans:
            if key not in key2index:
                key2index[key] = len(key2index) + 1
        return list(map(lambda x: key2index[x], key_ans))

    data1 = pd.read_csv("../../../../rllm/datasets/rel-movielens1m/regression/ratings/train.csv")
    data2 = pd.read_csv("../../../../rllm/datasets/rel-movielens1m/regression/ratings/test.csv")
    data = pd.concat([data1, data2], axis=0)
    sparse_features = ["MovieID", "UserID"]
    # 离散变量编码
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    # 离散特征和序列特征处理
    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=4) for feat in sparse_features]
    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns
    # 生成训练样本
    train_X, train_y = build_input_array(data, linear_feature_columns+dnn_feature_columns, target=['Rating'])
    test_X = torch.Tensor(train_X[797759:])
    train_X = torch.tensor(train_X[:797759])
    test_y = torch.Tensor(train_y[797759:])
    train_y = torch.tensor(train_y[:797759])
    return train_X, train_y, test_X, test_y, linear_feature_columns, dnn_feature_columns

def evaluate_data(model, dataloader):
    y_trues, y_preds = [], []
    for X, y in dataloader:
        y_trues.extend(y)
        y_preds.extend(model.predict(X))
    return mean_squared_error(y_trues, y_preds)


if __name__ == "__main__":
    # 加载数据集
    train_X, train_y, test_X, test_y, linear_feature_columns, dnn_feature_columns = get_data()
    train_dataloader = DataLoader(TensorDataset(train_X, train_y), batch_size=1024, shuffle=True) 
    test_dataloader = DataLoader(TensorDataset(test_X, test_y), batch_size=97384, shuffle=True) 
    # 模型定义
    model = DeepFM(linear_feature_columns, dnn_feature_columns)
    test_model = DeepFM(linear_feature_columns, dnn_feature_columns)
    model.to(device)
    model.compile(
        #loss=nn.MSELoss(),
        loss=nn.L1Loss(),#MAEloss
        optimizer=optim.Adam(model.parameters()),
        #metrics=['mse']
        metrics=['MAE']
    )
    start_time = time.time()
    # train
    model.fit(train_dataloader, epochs=1, steps_per_epoch=None, callbacks=[])
    end_time = time.time()
    # test
    mae = evaluate_data(model, test_dataloader)
    print()
    print("time: ", end_time - start_time)
    print('test_mae: ', mae)