import argparse
import os.path as osp
import sys

import torch
import dashscope

sys.path.append("../")

from examples.bridge.bridge_ifmain import data_prepare, train_bridge_model
from rllm.datasets import TLF2KDataset, TACM12KDataset, TML1MDataset

from annotation.annotation import annotate
from node_selection.node_selection import active_generate_mask
from langchain_community.llms import Tongyi
from utils import preprocess

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="tlf2k", choices=["tlf2k", "tml1m", "tacm12k"], help="dataset")
parser.add_argument("--hidden_channels", type=int, default=64, help="number of hidden channels in GCN")
parser.add_argument("--active_method", type=str, default="Degree", choices=["Random", "Degree"], help="active node selection")
parser.add_argument("--weighted_loss", type=bool, default=False, help="use weighted loss")
parser.add_argument("--n_tries", type=int, default=1, help="number of tries asking LLM")
parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--wd", type=float, default=1e-4, help="Weight decay")
parser.add_argument("--use_cache", type=bool, default=True, help="use cache")
parser.add_argument("--val", type=bool, default=True, help="use validation set")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

llm = Tongyi(dashscope_api_key='sk-19964643d8104217af9c6ee0802d446e', model_name='qwen-max-2025-01-25', model_kwargs={}, client=dashscope.Generation)

path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data")
if args.dataset == "TLF2K" or args.dataset == "tlf2k":
    dataset = TLF2KDataset(cached_dir=path, force_reload=True)
    train_num = 220
elif args.dataset == "TML1M" or args.dataset == "tml1m":
    dataset = TML1MDataset(cached_dir=path, force_reload=True)
    train_num = 140
elif args.dataset == "TACM12K" or args.dataset == "tacm12k":
    dataset = TACM12KDataset(cached_dir=path, force_reload=True)
    train_num = 280
else:
    print("invalid dataset")

target_table, user_embeddings, adj, rl, num_classes, emb_size, _, val_mask, test_mask = data_prepare(dataset, args.dataset, device)

processed_data = preprocess(dataset.data_list, args.dataset)
train_mask = active_generate_mask(processed_data, method=args.active_method, select_mask=~(test_mask | val_mask), train_num=train_num)
val_mask = val_mask.to("cpu")
test_mask = test_mask.to("cpu")

pl, _ = annotate(args.dataset, processed_data, train_mask | val_mask, llm, use_cache=args.use_cache, n_tries=args.n_tries)
y = pl.long().to(device)
y[test_mask] = rl.long().to(device)[test_mask]

print(f"using dataset: {args.dataset}, active method: {args.active_method}")

train_bridge_model(target_table, user_embeddings, adj, y, num_classes, emb_size, train_mask, val_mask, test_mask, args.epochs, args.lr, args.wd, device)