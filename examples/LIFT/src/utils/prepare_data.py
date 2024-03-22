import sys
sys.path.append("./")
sys.path.append("./../")
import numpy as np
# from baselines import clf_model
import utils.configs as cfgs
from utils.helper import log
from utils.classification_data_generator import DataGenerator, load_openml, df2jsonl
from sklearn.model_selection import StratifiedShuffleSplit
from utils.feature_names import df2jsonl_feat_name
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import pdb


# ####### Data
def mixup_data(y, X_raw, X_norm):
    lam = np.random.randint(11, size=y.shape) # 0 to 10, representing 10 * \lambda
    lam_ = lam.reshape(-1,1)
    idx = np.random.permutation(y.shape[0])

    X_raw_perm = X_raw[idx]
    X_norm_perm = X_norm[idx]

    X_raw_new = X_raw * lam_/10 + X_raw_perm * (1-lam_/10)
    X_norm_new = X_norm * lam_/10 + X_norm_perm * (1-lam_/10)

    y_perm = y[idx]
    y_new = y * lam + y_perm * (10-lam)
    assert(y_new.max() <= 10)
    assert(y_new.min() >= 0)

    print(y.shape, X_raw.shape, X_norm.shape)
    print(y_new.shape, X_raw_new.shape, X_norm_new.shape)

    return y_new, X_raw_new, X_norm_new


def prepare_data(did, context=False, mixup=0):
    data_gen = DataGenerator(did)
    if did < 10:
        y, X, _, att_names = data_gen.load_synthetic_datatsets(did)
        X_raw, X_norm = X, X
    else:
        y, X_raw, X_norm, att_names = load_openml(did)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

    # from IPython import embed; embed()
    count = 0
    for dev_index, test_index in sss.split(X_raw, y):
        assert count ==0
        X_raw_dev, X_raw_test = X_raw[dev_index], X_raw[test_index]
        y_dev, y_test = y[dev_index], y[test_index]
        X_norm_dev, X_norm_test = X_norm[dev_index], X_norm[test_index]
        count +=1
    if mixup:
        y_test *= 10

    # n_dev = len(dev_index)
    # train_index  = train_index[:int(.8*n_dev)]
    # val_index =  train_index[int(.8*n_dev):]
    # y_train, y_val, y_test =  y[train_index], y[val_index], y[test_index]
    # X_raw_train, X_raw_val, X_raw_test = X_raw[train_index], X_raw[val_index], X_raw[test_index]
    # X_norm_train, X_norm_val, X_norm_test = X_norm[train_index], X_norm[val_index], X_norm[test_index]
    # data = {'y_train': y_train, 'y_val': y_val, 'y_test': y_test, 'X_raw_train': X_raw_train, 'X_raw_test': X_raw_test, 'X_raw_val': X_raw_val, 'X_norm_test': X_norm_test, 'X_norm_val': X_norm_val, 'X_norm_train': X_norm_train, 'att_names': att_names}
    data = {'y_dev': y_dev, 'y_test': y_test, 'X_raw_dev': X_raw_dev, 'X_raw_test': X_raw_test, 'X_norm_test': X_norm_test, 'X_norm_dev': X_norm_dev}
    if mixup:
        np.save(f'data/{did}_dev_test_split_mixup', data)
    else:
        np.save(f'data/{did}_dev_test_split', data)

    # convert to prompt
    for i in range(3):
        count = 0
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=i)
        for train_index, val_index in sss.split(X_raw_dev, y_dev):
            assert count ==0
            train_index, val_index = train_index, val_index
            count += 1
        y_train, y_val = y_dev[train_index], y_dev[val_index]
        X_raw_train, X_raw_val = X_raw_dev[train_index], X_raw_dev[val_index]
        X_norm_train, X_norm_val = X_norm_dev[train_index], X_norm_dev[val_index]
        if mixup:
            y_train, X_raw_train, X_norm_train = mixup_data(y_train, X_raw_train, X_norm_train)
            y_val *= 10

        # save datasets
        data_i = {'y_train': y_train, 'y_val': y_val, 'y_test': y_test, 'X_raw_train': X_raw_train, 'X_raw_test': X_raw_test,
            'X_raw_val': X_raw_val, 'X_norm_test': X_norm_test, 'X_norm_val': X_norm_val, 'X_norm_train': X_norm_train, 'att_names': att_names}
        if mixup:
            np.save(f'data/{did}_train_val_test_split{i}_mixup', data_i)
        else:
            np.save(f'data/{did}_train_val_test_split{i}', data_i)

        train_df, val_df, test_df = pd.DataFrame(X_raw_train), pd.DataFrame(X_raw_val), pd.DataFrame(X_raw_test)
        train_df['y'], val_df['y'], test_df['y'] = y_train, y_val, y_test
        dfs = {'train': train_df, 'val': val_df, 'test': test_df}

        target_names = att_names[-1] if did > 10 else None
        feature_names = att_names[:-1] if did > 10 else None
        fname = f"{did}"
        jsonl_files = {}
        for mode in ['train', 'val', 'test']:
            if mixup:
                json_name = f'{fname}_split{i}_{mode}_context_{context}_mixup.jsonl'
            else:
                json_name = f'{fname}_split{i}_{mode}_context_{context}.jsonl'
            jsonl_files[mode] = df2jsonl(dfs[mode], json_name,
                            context = context,
                            feature_names = feature_names,
                            target_names = target_names,
                            init = 'Given that',
                            end = 'What is the category?')
            if did == 23 or did == 48 or did == 54:
                json_name = f'{fname}_split{i}_{mode}_context_{context}_feature_names.jsonl'
                jsonl_files[mode] = df2jsonl_feat_name(dfs[mode], json_name,did)


    print('Done', did)


if __name__ == '__main__':
    for did in [23,54]:
        prepare_data(did)
