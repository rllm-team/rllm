import sys 
sys.path.append("../../rllm/data")

import pandas as pd
import torch

import datatensor

def load():
    net_path = '../../rllm/datasets/titanic/'
    df = pd.read_csv(net_path + 'titanic.csv',
                     sep=',',
                     engine='python',
                     encoding='ISO-8859-1')
    
    y = torch.tensor(df['Survived'].values)
    df = df.drop(['PassengerId', 'Survived'], axis=1)
    x = df

    dataset = datatensor.legacy_init([x],
                ['v'],
                [y],
                ['v'],
                [],
                [])

    return dataset