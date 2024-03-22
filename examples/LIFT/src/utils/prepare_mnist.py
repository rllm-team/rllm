
import sys
sys.path.append('./')
sys.path.append('./../')
import os
import numpy as np
import torch
import random
import pandas as pd
import pdb

from utils import mnist
from utils.helper import write_jsonl, df2propmts, data2text
from utils.attack import get_adv_ex, add_noise

import argparse
parser = argparse.ArgumentParser(description='mnist data prepare')
parser.add_argument("-a", "--adv", default=0, type=int)
parser.add_argument("-e", "--eps", default=0.0, type=float)
parser.add_argument("-n", "--noisy", action="store_true")
parser.add_argument("-t", "--type", default='', type=str, choices=['const', 'unif', 'normal', 'sign', ''])
parser.add_argument("-p", "--is_permuted", action="store_true")
parser.add_argument("-s", "--sigma", default=0.0, type=float)
parser.add_argument("--source", default='lenet', type=str)
parser.add_argument("--target", default='mlp', type=str)
parser.add_argument("--noisy_train", default=0, type=int)
#parser.add_argument("-t", "--mlp_transfer", default=0, type=int)

args = parser.parse_args()
print(args)

##### Setting ########
random.seed(3247)
data_name = 'mnist' #'cifar10'
is_permuted = args.is_permuted
is_adv = args.adv #True # #True
is_noisy = args.noisy
#is_uniform = args.uniform
npy_stored = False #True
permuted = 'permuted_' if is_permuted else ''
eps_str = str(args.eps).replace('.', '_')
sigma_str = str(args.sigma).replace('.', '_')
source_str = f'_{args.source}'

if data_name == 'mnist':    
    n_train, n_val = 60000, 10000
else:
    n_train, n_val = 50000, 10000

###################
def reshape_mnist(x, is_adv=False):
    if is_adv:
        x = x.reshape(-1, 32, 32)
        x = x[:, 2:-2, 2:-2]
    else:
        x = x.reshape(-1, 28, 28)
    x = x[:,5:-5,5:-5]
    x = x.reshape(-1, 18*18)
# 		x = x > 255//2
    print(x.shape)
    return x

def reshape_cifar10(x):
    x = x.reshape(-1, 3, 32, 32)
    x = x[:,3, 5:-5,5:-5]
    x = x.reshape(-1, 18*18)
# 		x = x > 255//2
    return x

def load_npy(data='mnist', is_adv=True, eps=0.3):
    if data.lower() != 'mnist' or not is_adv:
        raise NotImplementedError

    if is_adv:
        eps_str = f'_{eps}'.replace('.', '_')
        with open('mnist_X_train.npy', 'rb') as f:
            X_train = np.load(f)
        with open('mnist_y_train.npy', 'rb') as f:
            y_train = np.load(f)
        with open(f'mnist_adv_X_test{eps_str}.npy', 'rb') as f:
            X_test = np.load(f)
        with open(f'mnist_adv_y_test{eps_str}.npy', 'rb') as f:
            y_test = np.load(f)
    else:
        with open('mnist_X_train.npy', 'rb') as f:
            X_train = np.load(f)
        with open('mnist_y_train.npy', 'rb') as f:
            y_train = np.load(f)
        with open('mnist_X_test.npy', 'rb') as f:
            X_test = np.load(f)
        with open('mnist_y_test.npy', 'rb') as f:
            y_test = np.load(f)
    
    return X_train, y_train, X_test, y_test


###################
# load data
if is_adv:
    if npy_stored:
        X, y, X_test, y_test = load_npy(data='mnist', is_adv=is_adv)
    else:
        X, y, X_test, y_test = get_adv_ex(data='mnist', pretrained=True, eps=args.eps, source=args.source, target=args.target)#test_mlp_attack=args.mlp_transfer)
else:
    X, y, X_test, y_test = mnist.load(data_name)
#pdb.set_trace()
if is_noisy:
    X_test = add_noise(X_test, sigma=args.sigma, eps=args.eps, noise_type=args.type) #is_uniform=is_uniform)
    if args.noisy_train:
        X = add_noise(X, sigma=args.sigma, eps=args.eps, noise_type=args.type)

# # do transfer learning on other neural network if necessary
# if args.mlp_transfer:
#     transfer_attack(X, y, X_test,y_test, model='mlp')
#     pdb.set_trace()
#     exit()

# permute
if is_permuted:
    permutation = np.arange(X.shape[1])
    random.shuffle(permutation)
    X = np.apply_along_axis(lambda x: x[permutation], 1, X)
    X_test = np.apply_along_axis(lambda x: x[permutation], 1, X_test)

# reshape 
if data_name == 'cifa10':
    X = reshape_cifar10(X)
    X_test = reshape_cifar10(X_test)
else:
    X = reshape_mnist(X, is_adv)
    X_test = reshape_mnist(X_test, is_adv)

# split data
idx = np.arange(X.shape[0])
random.shuffle(idx)
n_train = int(X.shape[0] * 0.8)
train_idx, val_idx = idx[:n_train], idx[n_train:]
X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y[train_idx], y[val_idx]

# dataframe
train_df, val_df, test_df = pd.DataFrame(X_train), pd.DataFrame(X_val), pd.DataFrame(X_test)
train_df['y'], val_df['y'], test_df['y'] = y_train, y_val, y_test
# data = {'y_train': y_train, 'y_val': y_val, 'y_test': y_test, 'X_train': X_train, 'X_test': X_test, 'X_val': X_val}
# if is_adv:
#     np.save(f'data/{permuted}{data_name}_adv{source_str}_{eps_str}', data)
# elif is_noisy:
#     if args.type == 'const':
#         np.save(f'data/{permuted}{data_name}_noisy_const_{eps_str}', data)        
#     elif args.type == 'unif':
#         np.save(f'data/{permuted}{data_name}_noisy_uniform_{eps_str}', data)        
#     elif args.type == 'normal':
#         np.save(f'data/{permuted}{data_name}_noisy_normal_{eps_str}', data)
#     elif args.type == 'sign':
#         np.save(f'data/{permuted}{data_name}_noisy_sign_{eps_str}', data)
#     else:
#         raise NotImplementedError
# else:
#     np.save(f'data/{permuted}{data_name}', data)
# print('Save ', data_name)




# convert to prompt
print(f'Save prompts for {data_name}, epsilon={args.eps}, noise={args.sigma}')
init='Given image with pixels '
end = 'What is this digit?'
train_prompts = df2propmts(train_df, data2text, init, end)
val_prompts = df2propmts(val_df, data2text, init, end)
test_prompts = df2propmts(test_df, data2text, init, end)

if is_adv:
    train_js = write_jsonl('\n'.join(train_prompts), f'{permuted}{data_name}_train.jsonl')
    val_js = write_jsonl('\n'.join(val_prompts), f'{permuted}{data_name}_val.jsonl')
    test_js = write_jsonl('\n'.join(test_prompts), f'{permuted}{data_name}_adv{source_str}_{eps_str}_test.jsonl')
elif is_noisy:
    if not args.noisy_train:
        train_js = write_jsonl('\n'.join(train_prompts), f'{permuted}{data_name}_train.jsonl')
        val_js = write_jsonl('\n'.join(val_prompts), f'{permuted}{data_name}_val.jsonl')
    else:
        train_js = write_jsonl('\n'.join(train_prompts), f'{permuted}{data_name}_noisy_{args.type}_{eps_str}_train.jsonl')
        val_js = write_jsonl('\n'.join(val_prompts), f'{permuted}{data_name}_noisy_{args.type}_{eps_str}_val.jsonl')

    test_js = write_jsonl('\n'.join(test_prompts), f'{permuted}{data_name}_noisy_{args.type}_{eps_str}_test.jsonl')
    # elif args.type == 'unif':
    #     test_js = write_jsonl('\n'.join(test_prompts), f'{permuted}{data_name}_noisy_uniform_{eps_str}_test.jsonl')
    # elif args.type == 'normal':
    #     test_js = write_jsonl('\n'.join(test_prompts), f'{permuted}{data_name}_noisy_normal_{eps_str}_test.jsonl')
    # elif args.type == 'sign':
        # test_js = write_jsonl('\n'.join(test_prompts), f'{permuted}{data_name}_noisy_sign_{eps_str}_test.jsonl')
else:
    train_js = write_jsonl('\n'.join(train_prompts), f'{permuted}{data_name}_train.jsonl')
    val_js = write_jsonl('\n'.join(val_prompts), f'{permuted}{data_name}_val.jsonl')
    test_js = write_jsonl('\n'.join(test_prompts), f'{permuted}{data_name}_test.jsonl')
