import time
import argparse
import sys
import os.path as osp

import torch
import torch.nn.functional as F

sys.path.append("./")
sys.path.append("../")
from rllm.datasets import TACM12KDataset
from rllm.transforms.graph_transforms import GCNTransform
from rllm.transforms.table_transforms import TabTransformerTransform
from rllm.nn.conv.graph_conv import GCNConv
from rllm.nn.conv.table_conv import TabTransformerConv
from rllm.nn.models import BRIDGE, TableEncoder, GraphEncoder
from rllm.data.relationframe import Relation, RelationFrame


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--wd", type=float, default=5e-4, help="Weight decay")
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
    writings_table,
    paper_embeddings,
    _,
) = dataset.data_list
emb_size = paper_embeddings.size(1)
target_table = papers_table.to(device)
y = papers_table.y.long().to(device)
paper_embeddings = paper_embeddings.to(device)

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

rf = RelationFrame(tables, relations=rel_l)
my_fpkey_sampler = FPkeySampler(rf, tables[0])