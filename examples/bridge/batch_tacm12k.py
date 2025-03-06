# The BRIDGE method from the "rLLM: Relational Table Learning with LLMs" paper.
# ArXiv: https://arxiv.org/abs/2407.20157

# Datasets  TACM12K
# Acc       0.2272

import time
import argparse
import sys
import os.path as osp

import torch
import torch.nn.functional as F

sys.path.append("./")
sys.path.append("../")
from rllm.datasets import TACM12KDataset
from rllm.data import Relation, RelationFrame
from rllm.dataloader import EntryLoader
from rllm.sampler import FPkeySampler
from rllm.transforms.graph_transforms import GCNTransform
from rllm.transforms.table_transforms import TabTransformerTransform
from rllm.nn.conv.graph_conv import GCNConv
from rllm.nn.conv.table_conv import TabTransformerConv
from rllm.nn.models import BRIDGE, TableEncoder, GraphEncoder
from utils import build_batch_homo_graph


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32, help="Dataloader batch size")
parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--wd", type=float, default=5e-4, help="Weight decay")
args = parser.parse_args()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# Load dataset
path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data")
dataset = TACM12KDataset(cached_dir=path, force_reload=True)

# Get the required data
(
    papers_table,
    authors_table,
    citations_table,
    writings_table,
    paper_embeddings,
    _,
) = dataset.data_list
emb_size = paper_embeddings.size(1)
papers_table = papers_table.to(device)
# y = papers_table.y.long().to(device)
paper_embeddings = paper_embeddings.to(device)

# Transform data
table_transform = TabTransformerTransform(
    out_dim=emb_size, metadata=papers_table.metadata
)
papers_table = table_transform(data=papers_table)

graph_transform = GCNTransform()

# Build relationframe
papers_table.table_name = "papers_table"
authors_table.table_name = "authors_table"
citations_table.table_name = "citations_table"
writings_table.table_name = "writings_table"

rel1 = Relation(fkey_table=citations_table, fkey="paper_id", pkey_table=papers_table, pkey="paper_id")
rel2 = Relation(fkey_table=citations_table, fkey="paper_id_cited", pkey_table=papers_table, pkey="paper_id")
rel3 = Relation(fkey_table=writings_table, fkey="paper_id", pkey_table=papers_table, pkey="paper_id")
rel4 = Relation(fkey_table=writings_table, fkey="author_id", pkey_table=authors_table, pkey="author_id")

tables = [papers_table, authors_table, citations_table, writings_table]
rel_l = [rel1, rel2, rel3, rel4]

f_p_path = [(papers_table, citations_table, rel1),
            (citations_table, papers_table, rel2),
            (writings_table, papers_table, rel3),
            (writings_table, authors_table, rel4)]

rf = RelationFrame(tables, relations=rel_l)
(
    train_loader,
    val_loader,
    test_loader,
) = EntryLoader.create(
    [papers_table.train_mask, papers_table.val_mask, papers_table.test_mask],
    seed_table=papers_table,
    sampling=True,
    rf=rf,
    Sampler=FPkeySampler,
    batch_size=args.batch_size,
    f_p_path=f_p_path,
)

# Set up model and optimizer
t_encoder = TableEncoder(
    in_dim=emb_size,
    out_dim=emb_size,
    table_conv=TabTransformerConv,
    metadata=papers_table.metadata,
)
g_encoder = GraphEncoder(
    in_dim=emb_size,
    out_dim=papers_table.num_classes,
    graph_conv=GCNConv,
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


def train():
    model.train()
    total_loss = .0
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        # For batch training, we build graph here.
        c_p_block = batch.dict_blocks[str(rel1)]
        p_c_block = batch.dict_blocks[str(rel2)]

        graph = build_batch_homo_graph((c_p_block, p_c_block), batch.target_table).to(device)
        adj = graph_transform(data=graph).adj
        logits = model(
            table=batch.target_table,
            non_table=paper_embeddings[torch.tensor(batch.target_table.oind, dtype=torch.long)],
            adj=adj
        )
        loss = F.cross_entropy(logits[batch.target_index],
                               batch.y.long().to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


@torch.no_grad()
def test(loader):
    model.eval()
    total_acc = 0.0
    for batch in loader:
        c_p_block = batch.dict_blocks[str(rel1)]
        p_c_block = batch.dict_blocks[str(rel2)]

        graph = build_batch_homo_graph((c_p_block, p_c_block), batch.target_table).to(device)
        adj = graph_transform(data=graph).adj
        logits = model(
            table=batch.target_table,
            non_table=paper_embeddings[torch.tensor(batch.target_table.oind, dtype=torch.long)],
            adj=adj
        )
        y = batch.y.long().to(device)
        total_acc += float(logits[batch.target_index].argmax(dim=1).eq(y).sum().item()) / y.size(0)
    return total_acc / len(loader)


start_time = time.time()
best_val_acc = best_test_acc = 0
for epoch in range(1, args.epochs + 1):
    train_loss = train()
    train_acc, val_acc, test_acc = test(train_loader), test(val_loader), test(test_loader)
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
