import sys 
sys.path.append("../../rllm/data")

import pandas as pd
import torch

import data

def load():
    net_path = '../../rllm/dataset/titanic/'
    df = pd.read_csv(net_path + 'titanic.csv',
                     sep=',',
                     engine='python',
                     encoding='ISO-8859-1')
    
    y = torch.tensor(df['Survived'].values)
    df = df.drop(['PassengerId', 'Survived'], axis=1)
    x = df

    dataset = data.DataLoader([x],
                ['v'],
                [y],
                ['v'],
                [],
                [])

    return dataset