import argparse
import os.path as osp
import sys

import torch
import dashscope

sys.path.append("../")

from examples.bridge.bridge_ifmain import data_prepare, train_bridge_model
from rllm.datasets import TLF2KDataset, TACM12KDataset, TML1MDataset
from rllm.nn.conv.graph_conv import GCNConv
from rllm.nn.conv.table_conv import TabTransformerConv
from rllm.nn.models import BRIDGE, TableEncoder, GraphEncoder

from annotation.annotation import annotate
from node_selection.node_selection import generate_mask
from langchain_community.llms import Tongyi
from utils_annotation_data import annotation_data_prepare

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="tlf2k", choices=["tlf2k", "tml1m", "tacm12k"], help="dataset")
parser.add_argument("--selection_method", type=str, default="Degree", choices=["Random", "Degree"], help="node selection method")
parser.add_argument("--n_tries", type=int, default=1, help="number of tries asking LLM")
parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--wd", type=float, default=1e-4, help="Weight decay")
parser.add_argument("--use_cache", type=bool, default=False, help="use cache")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

llm = Tongyi(dashscope_api_key="your-secret-key", model_name="qwen-max-2025-01-25", model_kwargs={}, client=dashscope.Generation)

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

annotation_data = annotation_data_prepare(dataset.data_list, args.dataset)
train_mask = generate_mask(annotation_data, method=args.selection_method, select_mask=~(test_mask | val_mask), train_num=train_num)
val_mask = val_mask.to("cpu")
test_mask = test_mask.to("cpu")

pl, _ = annotate(args.dataset, annotation_data, train_mask | val_mask, llm, use_cache=args.use_cache, n_tries=args.n_tries)
y = pl.long().to(device)
y[test_mask] = rl.long().to(device)[test_mask]

print(f"using dataset: {args.dataset}, selection method: {args.selection_method}")

t_encoder = TableEncoder(
    in_dim=emb_size,
    out_dim=emb_size,
    table_conv=TabTransformerConv,
    metadata=target_table.metadata,
)
g_encoder = GraphEncoder(
    in_dim=emb_size,
    out_dim=num_classes,
    graph_conv=GCNConv,
)
model = BRIDGE(
    table_encoder=t_encoder,
    graph_encoder=g_encoder,
).to(device)

train_bridge_model(model, target_table, user_embeddings, adj, y, num_classes, emb_size, train_mask, val_mask, test_mask, args.epochs, args.lr, args.wd, device)