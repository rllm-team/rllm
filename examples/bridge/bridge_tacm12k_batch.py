# The BRIDGE method from the "rLLM: Relational Table Learning with LLMs" paper.
# ArXiv: https://arxiv.org/abs/2407.20157

# Datasets          TACM12K
# Acc               0.305
# Full Batch        0.293

import time
import argparse
import sys
import os.path as osp

import torch
import torch.nn.functional as F

sys.path.append("./")
sys.path.append("../")
from rllm.datasets import TACM12KDataset
from rllm.dataloader import BRIDGELoader
from rllm.transforms.graph_transforms import NormalizeFeatures
from rllm.transforms.table_transforms import TabTransformerTransform
from rllm.nn.conv.graph_conv import GCNConv
from rllm.nn.conv.table_conv import TabTransformerConv
from rllm.nn.models import BRIDGE, TableEncoder, GraphEncoder
from utils import build_homo_graph


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--wd", type=float, default=5e-4, help="Weight decay")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
args = parser.parse_args()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data")
dataset = TACM12KDataset(cached_dir=path, force_reload=True)

# Get the required data
(
    papers_table,
    authors_table,
    citations_table,
    _,
    paper_embeddings,
    _,
) = dataset.data_list
emb_size = paper_embeddings.size(1)
target_table = papers_table.to(device)
y = papers_table.y.long().to(device)
paper_embeddings = paper_embeddings.to(device)

# Build graph
graph = build_homo_graph(
    relation_df=citations_table.df,
    n_all=len(papers_table),
).to(device)

# Transform data
table_transform = TabTransformerTransform(
    out_dim=emb_size, metadata=target_table.metadata
)
target_table = table_transform(data=target_table)
graph_transform = NormalizeFeatures()
graph = graph_transform(data=graph)

# Split data
train_mask, val_mask, test_mask = (
    target_table.train_mask,
    target_table.val_mask,
    target_table.test_mask,
)

# Dataloader
train_loader = BRIDGELoader(
    table=target_table,
    non_table=None,
    graph=graph,
    num_samples=[10, 5],
    train_mask=train_mask,
    batch_size=args.batch_size,
    shuffle=False,
)

# Set up model and optimizer
t_encoder = TableEncoder(
    in_dim=emb_size,
    out_dim=emb_size,
    table_conv=TabTransformerConv,
    metadata=target_table.metadata,
)
g_encoder = GraphEncoder(
    in_dim=emb_size,
    out_dim=target_table.num_classes,
    graph_conv=GCNConv,
    norm=True
)
model = BRIDGE(
    table_encoder=t_encoder,
    graph_encoder=g_encoder,
).to(device)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.wd,
)


def train() -> float:
    model.train()
    loss_all = 0
    for batch, _, adjs, table_data, _ in train_loader:
        optimizer.zero_grad()
        logits = model(
            table=table_data,
            non_table=None,
            adj=adjs,
        )
        loss = F.cross_entropy(
            logits[:batch], table_data.y[:batch].to(torch.long)
        )
        loss.backward()
        optimizer.step()
        loss_all += loss.item()
    return loss_all / len(train_loader)


@torch.no_grad()
def test():
    model.eval()
    logits = model(
        table=target_table,
        non_table=None,
        adj=graph.adj,
    )
    preds = logits.argmax(dim=1)

    accs = []
    for mask in [train_mask, val_mask, test_mask]:
        correct = float(preds[mask].eq(y[mask]).sum().item())
        accs.append(correct / int(mask.sum()))
    return accs


start_time = time.time()
best_val_acc = best_test_acc = 0
for epoch in range(1, args.epochs + 1):
    train_loss = train()
    train_acc, val_acc, test_acc = test()
    print(
        f"Epoch: [{epoch}/{args.epochs}]"
        f"Loss: {train_loss:.4f} train_acc: {train_acc:.4f} "
        f"val_acc: {val_acc:.4f} test_acc: {test_acc:.4f} "
    )
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc

print(f"Total Time: {time.time() - start_time:.4f}s")
print(
    "BRIDGE result: "
    f"Best Val acc: {best_val_acc:.4f}, "
    f"Best Test acc: {best_test_acc:.4f}"
)
