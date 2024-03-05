# NODE(Neural Oblivious Decision Ensembles) for classification task
# Paper: Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data(ICLR 2019)
# Arxiv: https://arxiv.org/pdf/1909.06312.pdf
# Macro_f1: 0.19028, Micro_f1: 0.36180
# Runtime: 9.07s (on a single 32G GPU)
# Cost: N/A
# Usage: lib
import sys
import time
sys.path.insert(0, '..')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append("../../")
import lib
import torch, torch.nn as nn
import torch.nn.functional as F
from qhoptim.pyt import QHAdam
from tqdm import tqdm
# from IPython.display import clear_output

device = 'cuda' if torch.cuda.is_available() else 'cpu'

experiment_name = 'classification'
experiment_name = '{}_{}.{:0>2d}.{:0>2d}_{:0>2d}:{:0>2d}'.format(experiment_name, *time.gmtime()[:5])
print("experiment:", experiment_name)

start_time = time.time()

# read the data
data = lib.Dataset("movies", data_path="/lustre/home/acct-stu/stu219/RLLM/rllm/rllm/datasets/rel-movielens1m/classification", random_state=1337, quantile_transform=True, quantile_noise=1e-3)

num_features = data.X_train.shape[1]
num_classes = 18

model = nn.Sequential(
    lib.DenseBlock(num_features, layer_dim=32, num_layers=1, tree_dim=num_classes + 1, flatten_output=False,
                   depth=3, choice_function=lib.entmax15, bin_function=lib.entmoid15),
    lib.Lambda(lambda x: x[..., :num_classes].mean(dim=-2)),
).to(device)

with torch.no_grad():
    res = model(torch.as_tensor(data.X_train[:2000], device=device))  
    
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)


trainer = lib.Trainer(
    model=model, loss_function=F.cross_entropy,
    experiment_name=experiment_name,
    warm_start=False,
    Optimizer=QHAdam,
    optimizer_params=dict(nus=(0.7, 1.0), betas=(0.95, 0.998), lr = 0.1),
    verbose=True,
    n_last_checkpoints=5
)

loss_history, err_history = [], []
best_val_err = 1.0
best_step = 0
early_stopping_rounds = 100
report_frequency = 20

for batch in lib.iterate_minibatches(data.X_train, data.y_train, batch_size=512, 
                                                shuffle=True, epochs=float('inf')):
    metrics = trainer.train_on_batch(*batch, device=device)
    
    loss_history.append(metrics['loss'])

    if trainer.step % report_frequency == 0:
        trainer.save_checkpoint()
        trainer.average_checkpoints(out_tag='avg')
        trainer.load_checkpoint(tag='avg')
        err = trainer.evaluate_classification_error(
            data.X_valid, data.y_valid, device=device, batch_size=512)
        
        if err < best_val_err:
            best_val_err = err
            best_step = trainer.step
            trainer.save_checkpoint(tag='best')
        
        err_history.append(err)
        trainer.load_checkpoint()  # last
        trainer.remove_old_temp_checkpoints()
            
        # clear_output(True)
        
    if trainer.step > best_step + early_stopping_rounds:
        print('BREAK. There is no improvment for {} steps'.format(early_stopping_rounds))
        print("Best step: ", best_step)
        print("Best Val Error Rate: %0.5f" % (best_val_err))
        break

trainer.load_checkpoint(tag='best')
macro_f1_error_rate = trainer.evaluate_classification_error(data.X_test, data.y_test, device=device, batch_size=1024)
micro_f1_error_rate = trainer.evaluate_classification_error_micro(data.X_test, data.y_test, device=device, batch_size=1024)

end_time = time.time()
elapsed_time = end_time - start_time

print('Best step: ', trainer.step)
print("Test Macro_f1: %0.5f" % (1-macro_f1_error_rate))
print("Test Micro_f1: %0.5f" % (1-micro_f1_error_rate))
# trainer.load_checkpoint()

print(f"Total time: {elapsed_time} seconds")