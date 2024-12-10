# The RECT method from the
# "Network Embedding with Completely-imbalanced Labels" paper.
# ArXiv: https://arxiv.org/abs/2007.03545

# RECT focuses on the zero-shot setting,
# where only parts of classes have labeled samples.
# As such, "unseen" classes are first removed from the training set.
# Then we train a RECT (or more specifically its supervised part RECT-L) model.
# Lastly, we use the Logistic Regression model to evaluate the performance
# of the resulted embeddings based on the original balanced labels.

# Datasets              Citeseer    |         Cora         | Pubmed
# Unseen Classes  [1, 2, 5]  [3, 4] | [1, 2, 3]  [3, 4, 6] | [2]
# RECT-L          57.50      60.80  | 65.20      65.70     | 64.50

import argparse
import copy
import time
import sys
import os.path as osp

import torch
from sklearn.linear_model import LogisticRegression

sys.path.append("./")
sys.path.append("../")
from rllm.datasets import PlanetoidDataset
from rllm.nn.models import RECT_L, GNNConfig
from rllm.transforms.utils import RemoveTrainingClasses


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", type=str, default="cora", choices=["cora", "citeseer", "pubmed"]
)
parser.add_argument("--unseen-classes", type=int, nargs="*", default=[1, 2, 3])
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--wd", type=float, default=0, help="Weight decay")
parser.add_argument("--dropout", type=float, default=0.0, help="Graph Dropout")
parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
args = parser.parse_args()

# Load data
path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data")
dataset = PlanetoidDataset(path, args.dataset, force_reload=True)
data = dataset[0]

# Transform data
transform = GNNConfig.get_transform("RECT")()
data = transform(data)

zs_data = RemoveTrainingClasses(args.unseen_classes)(copy.deepcopy(data))

model = RECT_L(200, 200, dropout=args.dropout)
zs_data.y = model.get_semantic_labels(zs_data.x, zs_data.y, zs_data.train_mask)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model, zs_data = model.to(device), zs_data.to(device)

criterion = torch.nn.MSELoss(reduction="sum")
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.wd,
)

model.train()
st = time.time()
for epoch in range(1, args.epochs + 1):
    optimizer.zero_grad()
    out = model(zs_data.x, zs_data.adj)
    loss = criterion(out[zs_data.train_mask], zs_data.y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch:03d}, Loss {loss:.4f}")
et = time.time()
model.eval()
with torch.no_grad():
    h = model.embed(zs_data.x, zs_data.adj).cpu()

reg = LogisticRegression()
reg.fit(h[data.train_mask].numpy(), data.y[data.train_mask].numpy())
test_acc = reg.score(h[data.test_mask].numpy(), data.y[data.test_mask].numpy())
print(f"Total Time  : {et-st:.4f}s")
print(f"Test Acc    : {test_acc:.4f}")
