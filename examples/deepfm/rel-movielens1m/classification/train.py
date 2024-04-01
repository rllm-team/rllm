# Deepfm for classification task 
# Paper: DeepFM: A Factorization-Machine based Neural Network for CTR Prediction
# arxiv : https://arxiv.org/abs/1703.04247
# macro_f1: 0.193, micro_f1:0.395
# Runtime: 8.07s on CPU with 10 cores(Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz)
# Cost: N/A
# Description: Paper Reproduction.

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
import sys
sys.path.append("../../src")
from inputs import DenseFeat, SparseFeat, VarLenSparseFeat, build_input_array, TensorDataset
from models import DeepFM
from snippets import sequence_padding, seed_everything
from callbacks import Evaluator
from sklearn.metrics import f1_score
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed_everything(42)

def get_data():
    def split(x):
        key_ans = x.split('|')
        for key in key_ans:
            if key not in key2index:
                key2index[key] = len(key2index) + 1
        return list(map(lambda x: key2index[x], key_ans))

    data1 = pd.read_csv("../../../../rllm/datasets/rel-movielens1m/classification/movies/train.csv")
    data2 = pd.read_csv("../../../../rllm/datasets/rel-movielens1m/classification/movies/test.csv")
    data = pd.concat([data1, data2], axis=0)
    #data['rating'] = data['rating'] - 1
    sparse_features = ["MovielensID","Year","Director","Title","Cast"]

    # 离散变量编码
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    # 序列特征处理
    key2index = {}
    genres_list = list(map(split, data['Genre'].values))
    # 保证最大长度在6列，但不影响数据
    max_genres = 0
    index = 0
    for i in range(len(genres_list)):
        if len(genres_list[i]) > max_genres:
            max_genres = len(genres_list[i])
            index = i
    if max_genres < 6:
        pad = [0 for i in range(6-max_genres)]
        genres_list[index].extend(pad)
    genres_list = sequence_padding(genres_list)
    
    data['Genre'] = genres_list.tolist()
    print()

    # 离散特征和序列特征处理
    # 对于稀疏特征，通过嵌入技术将他们转换为密集向量。对于密集的数值特征，将它们连接到全连接层的输入张量。
    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=4) for feat in sparse_features]
    varlen_feature_columns = [VarLenSparseFeat(SparseFeat('Genre', vocabulary_size=len(
        key2index) + 1, embedding_dim=4), maxlen=genres_list.shape[-1], pooling='mean')]
    #生成特征列
    linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns
    
    # 生成训练样本
    train_X, train_y = build_input_array(data, linear_feature_columns+dnn_feature_columns, target='Genre')
    test_X = torch.tensor(train_X[389:])
    train_X = torch.tensor(train_X[:389])
    floated = [i for i in train_y]
    for i in range(len(train_y)):
        for j in range(len(train_y[i])):
            floated[i][j] = float(train_y[i][j])
    updated_floated = [[0]*18 for i in range(len(floated))]
    for i in range(len(floated)):
        for j in floated[i]:
            if j != 0:
                updated_floated[i][int(j-1)] = float(1)
    train_y = torch.tensor(updated_floated[:389])
    test_y = torch.tensor(updated_floated[389:])
    return train_X, train_y, test_X, test_y, linear_feature_columns, dnn_feature_columns

def evaluate_data(model, dataloader):
    y_preds = []
    for X, y in dataloader:
        y_prob = model.predict(X).cpu().numpy()
        y = y.cpu().numpy()
        y_preds = (y_prob > 0).astype(int)
    return f1_score(y, y_preds, average='micro', zero_division=1), f1_score(y, y_preds, average='macro', zero_division=1)

class MyEvaluator(Evaluator):
    def evaluate(self):
        micro, macro = evaluate_data(self.model, train_dataloader)
        return {'micro_f1': micro, 'macro_f1': macro}


if __name__ == "__main__":
    # 加载数据集
    train_X, train_y, test_X, test_y, linear_feature_columns, dnn_feature_columns = get_data()
    train_dataloader = DataLoader(TensorDataset(train_X, train_y, device=device), batch_size=256, shuffle=True) 
    test_dataloader = DataLoader(TensorDataset(test_X, test_y, device=device), batch_size=3108, shuffle=True) 

    # 模型定义
    model = DeepFM(linear_feature_columns, dnn_feature_columns, out_dim=18)
    test_model = DeepFM(linear_feature_columns, dnn_feature_columns, out_dim=18)
    model.to(device)

    model.compile(
        #loss=nn.CrossEntropyLoss(),
        loss=nn.MSELoss(),
        #optimizer=optim.Adam(model.parameters(), lr=1e-2)
        optimizer=optim.Adam(model.parameters())
    )

    # 评价器定义
    evaluator1 = MyEvaluator(monitor='macro_f1')
    evaluator2 = MyEvaluator(monitor='micro_f1')
    
    start_time = time.time()
    # train
    model.fit(train_dataloader, epochs=200, steps_per_epoch=None, callbacks=[evaluator1, evaluator2])
    end_time = time.time()
    # test
    micro, macro = evaluate_data(model, test_dataloader)
    print()
    print("time: ", end_time - start_time)
    print('test_micro_f1: ', micro)
    print('test_macro_f1: ', macro)