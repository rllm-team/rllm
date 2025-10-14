## Train LMs
fine-tuning and save `TA` features:
```shell
WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=0,1,2,3 python ./examples/tape/train_lm.py dataset cora
```
fine-tuning and save `E` features:
```shell
WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=0,1,2,3 python ./examples/tape/train_lm.py dataset cora lm.train.use_gpt True
```

## Train GNNs
```shell
python ./examples/tape/train_gnn.py gnn.model.name GCN
```