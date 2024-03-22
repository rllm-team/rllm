import sys
sys.path.append('./')
sys.path.append('./../')
from re import A
import pandas as pd
import numpy as np
import openai
import argparse
import random
import time, json, os
from sklearn.utils import shuffle
from tqdm import tqdm

from utils.classification_data_generator import DataGenerator, df2jsonl,array2prompts
from utils.helper import log
import utils.configs as cfgs
from utils.prepare_data import prepare_data
from utils.corrupt_labels import corrupt_labels
from models.GPT3FineTuner import GPT3FineTuner

def get_in_context_prefix_prompts(jsonl_file,num_prompts):
    prefix_prompts = ''
    count = 0
    with open(jsonl_file) as fp:
        for i,line in enumerate(fp):
            count += 1
            if i >= num_prompts -1:
                break
            json_obj = json.loads(line)
            prefix_prompts += json_obj['prompt']
            prefix_prompts += json_obj['completion']
            prefix_prompts += ''
    print("Number of prefix prompts:",count)
    return prefix_prompts
    
def extract_completion(jsonl_file):
    completions = []
    with open(jsonl_file) as fp:
        for line in fp:
            json_obj = json.loads(line)
            completions.append(json_obj['completion'])
    return completions

def extract_subset(jsonl_file,num_prompts,random_state):
    training_set = []
    with open(jsonl_file) as fp:
        for j,line in enumerate(fp):
            training_set.append(line)
    training_set = pd.DataFrame(training_set)
    subfname = jsonl_file.split('.jsonl')[0]+"_subset.jsonl"
    shuffled = shuffle(training_set,random_state=random_state)
    shuffled = shuffled.values.tolist()
    subset = shuffled[:num_prompts]
    jsonl = ''.join(subset)
    
    with open(subfname, 'w') as f:
        f.write(jsonl)
    return subfname


def extract_random_incontext_prompts(jsonl_file:list,num_prompts,target_jsonl,random_state):
    prefix_prompts = []
    for jf in jsonl_file:
        with open(jf) as fp:
            for i,line in enumerate(fp):
                json_obj = json.loads(line)
                prefix_prompts.append(json_obj['prompt']+json_obj['completion']+' ')
    prefix_prompts = pd.DataFrame(prefix_prompts)
    in_context_prompts = []
    with open(target_jsonl) as fp:
        for line in fp:
            shuffled = shuffle(prefix_prompts,random_state=random_state)
            json_obj = json.loads(line)
            prefix_prompt = ''
            for j in range(num_prompts):
                prefix_prompt+=shuffled.values[j][0]
            in_context_prompts.append(prefix_prompt+json_obj['prompt'])
    # print(len(in_context_prompts),in_context_prompts[0])
    return in_context_prompts

def load_subset(jsonl_file:list,num_prompts,target_jsonl,random_state):
    prefix_prompts = []
    for jf in jsonl_file:
        with open(jf) as fp:
            for i,line in enumerate(fp):
                json_obj = json.loads(line)
                prefix_prompts.append(json_obj['prompt']+json_obj['completion']+' ')
    prefix_prompts = pd.DataFrame(prefix_prompts)
    in_context_prompts = []
    with open(target_jsonl) as fp:
        for line in fp:
            shuffled = shuffle(prefix_prompts,random_state=random_state)
            json_obj = json.loads(line)
            prefix_prompt = ''
            for j in range(num_prompts):
                prefix_prompt+=shuffled.values[j][0]
            in_context_prompts.append(prefix_prompt+json_obj['prompt'])
    # print(len(in_context_prompts),in_context_prompts[0])
    return in_context_prompts

    
def prompt2value(x):
    # print("Output:",x)
    c = x.strip().split('@@@')[0]
    if c == '':
        return None
    try:
        return int(c)
    except:
        return c

def load_data(did, index,mixup=False):
    if mixup:
        fname = f'data/{did}_dev_test_split_mixup.npy'    
    else:
        fname = f'data/{did}_dev_test_split.npy'
    print("dev-test split file name",fname)
    if not os.path.isfile(fname):
        print('prepare data', did)
        prepare_data(did, context=False, mixup=mixup)
    if mixup:
        ith_fname = f'data/{did}_train_val_test_split{index}_mixup.npy'
    else:
        ith_fname = f'data/{did}_train_val_test_split{index}.npy'
    print("current loading train-val-test file name",ith_fname)
    data = np.load(ith_fname, allow_pickle=True)
    data = data.item()
    X_train, X_val, X_test = data['X_norm_train'], data['X_norm_val'], data['X_norm_test']
    y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']
    return X_train, y_train, X_val, y_val, X_test, y_test

def load_corrupted_data(did, index, random_label_noise,type='random'):
    ith_fname = f'data/{did}_train_val_test_split{index}.npy'
    data = np.load(ith_fname, allow_pickle=True)
    data = data.item()
    X_train = data['X_raw_train']
    y_train = data['y_train']

    if type == 'random':
        corrupted_y_train = corrupt_labels(y_train, random_label_noise)
    elif type == 'system':
        corrupted_y_train = systematically_corrupt_labels(y_train, random_label_noise)
    else:
        raise NotImplementedError

    corrupted_train_df = pd.DataFrame(X_train)
    corrupted_train_df['y'] = corrupted_y_train

    # from IPython import embed; embed()
    json_name = f'{did}_split{index}_train_context_False_corrupted_{random_label_noise}.jsonl'
    return df2jsonl(corrupted_train_df, json_name,
                            context = False,
                            # feature_names = feature_names,
                            # target_names = target_names,
                            init = 'When we have',
                            end = 'What is this type?')

def systematically_corrupt_labels(y_train, random_label_noise=0.0):
    random.seed(42)
    np.random.seed(42)

    y_train = np.array(y_train, copy=True)
    possible_labels = np.unique(y_train)

    # Choose n random indices
    n_corrupted = int(len(y_train) * random_label_noise)
    random_noise_indices = np.random.choice(
        len(y_train), size=n_corrupted, replace=False)
    random_noise = np.full(y_train.shape, True)
    random_noise[random_noise_indices] = False

    # Increments label by 1 for systematic bias.
    # That means all errors for the same class will map to the same (different) corrupted class
    def gen_corrupt_label(_, i):
        i = int(i)
        return (y_train[i] + 1) % len(possible_labels)
    corrupted_labels = np.fromfunction(
        np.vectorize(gen_corrupt_label), (1, len(y_train)))[0]

    return np.where(random_noise, y_train, corrupted_labels)

def load_jsonl(did, idx, context=False,feature_name=False):
    jsonl_files = {}
    if feature_name:
        for mode in ['train', 'val', 'test']:
            jsonl_files[mode] = f'data/{did}_split{idx}_{mode}_context_{context}_feature_names.jsonl'
    else:
        for mode in ['train', 'val', 'test']:
            jsonl_files[mode] = f'data/{did}_split{idx}_{mode}_context_{context}.jsonl'
    return jsonl_files


def extract_prompts(jsonl_file,in_context_prefix=''):
    test_prompts = []
    with open(jsonl_file,encoding='utf-8') as fp:
        for line in fp:
            json_obj = json.loads(line)
            test_prompts.append(in_context_prefix+json_obj['prompt'])
    return test_prompts

def generate(gpt, text_lst, max_token=10, batch_size=2):
    gpt.model.eval()
    outputs = []
    for i in tqdm(np.arange(0, len(text_lst), batch_size)):
        texts = text_lst[i:min(i + batch_size, len(text_lst))]
        prompt = gpt.tokenizer(texts, truncation=True, padding = True, max_length=1024, return_tensors='pt')
        prompt = {key: value.to(gpt.device) for key, value in prompt.items()}
        outs = gpt.model.generate(**prompt, max_new_tokens=max_token, pad_token_id=gpt.tokenizer.eos_token_id, do_sample=True, early_stopping = True)
        outs = gpt.tokenizer.batch_decode(outs, skip_special_tokens=True)
        outputs += outs
    return outputs

def generate_output(gpt3_fine_tuner, val_prompts):
    ans, bs, count = [], 20, 0
    while count < len(val_prompts):
        start, end = count, min(count + bs, len(val_prompts))
        count = end
        batch = val_prompts[start:end]
        # print('Input:',batch)
        ans += gpt3_fine_tuner.query(batch)
    return [prompt2value(x) for x in ans]

def generate_output_in_context(prompts, use_model):
    ans, bs, count = [], 20, 0
    while count < len(prompts):
        start, end = count, min(count + bs, len(prompts))
        count = end
        batch = prompts[start:end]
        outputs = openai.Completion.create(model=use_model,prompt = batch, temperature=0)
        ans += [outputs['choices'][i]['text'] for i in range(len(batch))]
    return [prompt2value(x) for x in ans]

def get_accuracy(y_pred_val, y_val):
    acc_val = (y_pred_val == y_val).mean()
    acc_val = round(acc_val * 100, 2)
    return acc_val


def run_gpt3(did, jsonl_train, jsonl_val, val_prompts,test_prompts,in_context,openai_config,positive_class=None,):
    # operate_all_Files_on_openai(op='delete')
    gpt3_fine_tuner = GPT3FineTuner(openai_config, jsonl_train, jsonl_val)
    if cfgs.openml_data_ids[did] == 2:
        clf_cfgs = {'n_classes': 2, 'positive_class': f'{positive_class}@@@'}  
    else:
        clf_cfgs = {'n_classes': cfgs.openml_data_ids[did]}
    if in_context:
        y_pred_val = generate_output_in_context(val_prompts)
        y_pred = generate_output_in_context(test_prompts)
    else:
        gpt3_fine_tuner.fine_tune(clf_cfgs)
        y_pred_val = generate_output(gpt3_fine_tuner, val_prompts)
        y_pred = generate_output(gpt3_fine_tuner, test_prompts)
    return y_pred_val,y_pred,gpt3_fine_tuner


def query(gpt, prompts, bs=10):
    outputs = generate(gpt, prompts, batch_size=bs)
    ans = []
    for txt in outputs:
        try:
            output = prompt2value(txt.split('@@@')[0].split('###')[-1])
        except:
            output = txt
        ans.append(output)
    return ans