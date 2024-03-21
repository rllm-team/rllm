import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer, LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from category_encoders import LeaveOneOutEncoder
import sys 
import os
current_path = os.path.dirname(__file__)
sys.path.append(current_path + '/../data')

import pandas as pd
import numpy as np
import torch

def remove_unused_column(data):
    unused_list = []
    for col in data.columns:
        uni = len(data[col].unique())
        if uni <= 1:
            unused_list.append(col)
    data.drop(columns=unused_list, inplace=True)
    return data


def encode_class(train_ge, valid_ge, test_ge):
    y_train = np.array([x.split('|')[0] for x in train_ge])
    y_valid = np.array([x.split('|')[0] for x in valid_ge])
    y_test = np.array([x.split('|')[0] for x in test_ge])
    label_encoder = LabelEncoder()
    y_t = label_encoder.fit_transform(y_train)
    y_v = label_encoder.fit_transform(y_valid)
    y_s = label_encoder.fit_transform(y_test)
    return y_t, y_v, y_s


def split_data(data, target, test_size):
    label = data[target]
    data = data.drop([target], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=test_size, random_state=123, shuffle=True)
    return X_train, y_train.values, X_test, y_test.values


def quantile_transform(X_train, X_valid, X_test):
    quantile_train = np.copy(X_train)
    qt = QuantileTransformer(random_state=55688, output_distribution='normal').fit(quantile_train)
    X_train = qt.transform(X_train)
    X_valid = qt.transform(X_valid)
    X_test = qt.transform(X_test)
    return X_train, X_valid, X_test


def MSLR():
    target = 0
    train = pd.read_pickle('./data/MSLR-WEB10K/train.pkl')
    valid = pd.read_pickle('./data/MSLR-WEB10K/valid.pkl')
    test = pd.read_pickle('./data/MSLR-WEB10K/test.pkl')
    y_train = train[target].values
    y_valid = valid[target].values
    y_test = test[target].values
    train.drop([target], axis=1, inplace=True)
    valid.drop([target], axis=1, inplace=True)
    test.drop([target], axis=1, inplace=True)
    X_train, X_valid, X_test = quantile_transform(train, valid, test)

    return X_train, y_train, X_valid, y_valid, X_test, y_test



def cardio():
    target = 'cardio'
    data = pd.read_csv('./data/dataset/cardio/cardiovascular-disease.csv', delimiter=';').drop(['id'], axis=1)
    train = pd.read_csv('./data/cardio/train_idx.csv', delimiter=';')
    valid = pd.read_csv('./data/cardio/valid_idx.csv', delimiter=';')
    test = pd.read_csv('./data/cardio/test_idx.csv', delimiter=';')
    y_train = train[target].values
    train.drop([target], axis=1, inplace=True)
    y_valid = valid[target].values
    valid.drop([target], axis=1, inplace=True)
    y_test = test[target].values
    test.drop([target], axis=1, inplace=True)
    X_train, X_valid, X_test = quantile_transform(train, valid, test)
    return X_train, y_train, X_valid, y_valid, X_test, y_test

def movielens_regression():
    net_path = './data/movie_reg'
    target = 'Rating' 
    test = pd.read_pickle(net_path + '/test.pkl')
    train = pd.read_pickle(net_path + '/train.pkl')
    valid = pd.read_pickle(net_path + '/valid.pkl')
    y_train = train[target].values
    X_train = np.array(train.drop([target], axis=1))
    y_valid = valid[target].values
    X_valid = np.array(valid.drop([target], axis=1))
    y_test = test[target].values
    X_test = np.array(test.drop([target], axis=1))

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def movielens_classification():
    print("yes")
    net_path = './data/movie_cla'
    target = 'Genre'  
    test = pd.read_csv(net_path + '/test.csv')
    train = pd.read_csv(net_path + '/train.csv')
    valid = pd.read_csv(net_path + '/valid.csv')
    X_train = np.array(train[['Year', 'MovielensID']])
    X_valid = np.array(valid[['Year','MovielensID' ]])
    X_test = np.array(test[['Year', 'MovielensID']])
    train_genres = train['Genre']
    validation_genres = valid['Genre']
    test_genres = test['Genre']
    y1_train = train_genres.values
    y1_valid = validation_genres.values
    y1_test = test_genres.values
    y_train, y_valid, y_test = encode_class(y1_train, y1_valid, y1_test)

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def get_data(datasetname):
    if datasetname == 'MSLR':
        return MSLR()
    elif datasetname == 'cardio':
        return cardio()
    elif datasetname == 'movielens_reg':
        return movielens_regression()
    elif datasetname == 'movielens_cls':
        return movielens_classification()

if __name__ == '__main__':
    movielens_classification()

