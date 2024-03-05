# NODE(Neural Oblivious Decision Ensembles) for regression task
# Paper: Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data(ICLR 2019)
# Arxiv: https://arxiv.org/pdf/1909.06312.pdf
# MSE: 1.0292
# Runtime: 1256.8s (on a single 32G GPU)
# Cost: N/A
# Usage: lib

import os, sys
import time
sys.path.insert(0, '..')
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../../")
import lib
import torch, torch.nn as nn
import torch.nn.functional as F
from qhoptim.pyt import QHAdam
from tqdm import tqdm
# from IPython.display import clear_output

device = 'cuda' if torch.cuda.is_available() else 'cpu'

experiment_name = 'rating'
experiment_name = '{}_{}.{:0>2d}.{:0>2d}_{:0>2d}:{:0>2d}'.format(experiment_name, *time.gmtime()[:5])
print("experiment:", experiment_name)

start_time = time.time()

data = lib.Dataset("ratings", data_path="/lustre/home/acct-stu/stu219/RLLM/rllm/rllm/datasets/rel-movielens1m/regression",random_state=1337, quantile_transform=True, quantile_noise=1e-3)

in_features = data.X_train.shape[1]

mu, std = data.y_train.mean(), data.y_train.std()
normalize = lambda x: ((x - mu) / std).astype(np.float32)
data.y_train, data.y_valid, data.y_test = map(normalize, [data.y_train, data.y_valid, data.y_test])

print("mean = %.5f, std = %.5f" % (mu, std))

model = nn.Sequential(
    lib.DenseBlock(in_features, 128, num_layers=2, tree_dim=3, depth=6, flatten_output=False,
                   choice_function=lib.entmax15, bin_function=lib.entmoid15),
    lib.Lambda(lambda x: x[..., 0].mean(dim=-1)),  # average first channels of every tree
    
).to(device)

with torch.no_grad():
    res = model(torch.as_tensor(data.X_train[:5000], device=device))
    # trigger data-aware init
    
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)


optimizer_params = { 'nus':(0.7, 1.0), 'betas':(0.95, 0.998)}

trainer = lib.Trainer(
    model=model, loss_function=F.mse_loss,
    experiment_name=experiment_name,
    warm_start=False,
    Optimizer=QHAdam,
    optimizer_params=optimizer_params,
    verbose=True,
    n_last_checkpoints=5
)

loss_history, mse_history = [], []
best_mse = float('inf')
best_step_mse = 0
early_stopping_rounds = 5000
report_frequency = 200

for batch in lib.iterate_minibatches(data.X_train, data.y_train, batch_size=1024, 
                                                shuffle=True, epochs=float('inf')):
    metrics = trainer.train_on_batch(*batch, device=device)
    
    loss_history.append(metrics['loss'])

    if trainer.step % report_frequency == 0:
        trainer.save_checkpoint()
        trainer.average_checkpoints(out_tag='avg')
        trainer.load_checkpoint(tag='avg')
        mse = trainer.evaluate_mse(
            data.X_valid, data.y_valid, device=device, batch_size=16384)

        if mse < best_mse:
            best_mse = mse
            best_step_mse = trainer.step
            trainer.save_checkpoint(tag='best_mse')
        mse_history.append(mse)
        
        trainer.load_checkpoint()  # last
        trainer.remove_old_temp_checkpoints()

        # clear_output(True)

    if trainer.step > best_step_mse + early_stopping_rounds:
        print('BREAK. There is no improvment for {} steps'.format(early_stopping_rounds))
        print("Best step: ", best_step_mse)
        print("Best Val MSE: %0.5f" % (best_mse))
        break

trainer.load_checkpoint(tag='best_mse')
mse = trainer.evaluate_mse(data.X_test, data.y_test, device=device)
mae = trainer.evaluate_mae(data.X_test, data.y_test, device=device)

end_time = time.time()
elapsed_time = end_time - start_time
print('Best step: ', trainer.step)
print("Test MSE: %0.5f" % (mse))
print("Test MAE: %0.5f" % (mae))
print(f"Total time: {elapsed_time} seconds")