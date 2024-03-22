import pandas as pd
from functools import partial
import sys
import time

sys.path.append("../src")
from utils.helper import data2text,write_jsonl
import models.lora_gptj as GPTJ
from run_exps_helper import *
import torch
from sklearn.preprocessing import MultiLabelBinarizer

sys.path.append("../../../")
from rllm.utils import mae, get_llm_chat_cost

time_start = time.time()

def data2text(row, label = True, init = '', end = ''):
    prompt = init 
    prompt += ' UserID: ' + str(row['UserID']).replace("'", "").replace('"', '') +\
            ' MovieID: ' + str(row['MovieID']).replace("'", "").replace('"', '')
            # 'Movie Title: '+ str(movies.loc(movies['MovielensID'] == row['MovieID'])['Title'].replace("'", "").replace('"', ''))
    prompt += end

    if not label:
        final_prompt = f"{prompt}###"
    else:
        completion = row['Rating']
        final_prompt = "{\"prompt\":\"%s###\", \"completion\":\"%s@@@\"}" % (prompt, completion)
    return final_prompt

def df2propmts(df, data2text_func, init = '', end = ''):
    jsonl = df.apply(func = partial(data2text_func, init = init, end = end), axis = 1).tolist()
    return jsonl

parser = argparse.ArgumentParser(description='')
parser.add_argument("-g", "--gpu_id", default=0, type=int)
parser.add_argument("--local_rank", default=-1, type=int)
parser.add_argument("--seed", default=12345, type=int)
parser.add_argument("-p", "--is_permuted", action="store_true")

parser.add_argument("-v", "--eval", default=0, type=int)
args = parser.parse_args()

device = torch.device(f'cuda:{args.gpu_id}') if torch.cuda.is_available() else 'cpu'
torch.cuda.set_device(args.gpu_id)

users = pd.read_csv(
    '../../../rllm/datasets/rel-movielens1m/regression/users.csv')
train = pd.read_csv(
    '../../../rllm/datasets/rel-movielens1m/regression/ratings/train.csv')
val = pd.read_csv(
    '../../../rllm/datasets/rel-movielens1m/regression/ratings/validation.csv')
test = pd.read_csv(
    '../../../rllm/datasets/rel-movielens1m/regression/ratings/test.csv')
movies = pd.read_csv(
    '../../../rllm/datasets/rel-movielens1m/regression/movies.csv')

init= 'Given a UserID and a MovieID'
end = 'What\'s the rating that the user will give to the movie? Give a single number as rating without saying anything else. '

train_prompts = df2propmts(train, data2text, init, end)
val_prompts = df2propmts(val, data2text, init, end)
test_prompts = df2propmts(test, data2text, init, end)


write_jsonl('\n'.join(train_prompts),'train.json')
write_jsonl('\n'.join(val_prompts),'val.json')
write_jsonl('\n'.join(test_prompts),'test.json')

y_val = val['Rating']
y_test = test['Rating']





# gpt = GPTJ.LoRaQGPTJ(adapter=True, device=device,model_name='hivemind/gpt-j-6B-8bit')
gpt = GPTJ.LoRaQGPTJ(adapter=True, device=device)
train_configs={'learning_rate': 1e-5, 'batch_size': 1, 'epochs':1,  'weight_decay': 0.01, 'warmup_steps': 6}
gpt.finetune('data/train.json', 'data/val.json', train_configs, saving_checkpoint=True)

y_pred= [int(p) for p in query(gpt, test_prompts, bs=16)]


# acc = get_accuracy(y_pred, y_test)
# print(acc)

mae_loss = mae(y_test, y_pred)

time_end = time.time()

print(f"mae_loss: {mae_loss}")

print(f"Total time: {time_end - time_start}s")
# print(f"Total USD$: {total_cost}")