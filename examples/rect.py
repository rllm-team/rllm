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
import time
import sys
import os.path as osp
from copy import deepcopy

import torch
from sklearn.linear_model import LogisticRegression

sys.path.append("./")
sys.path.append("../")
from rllm.datasets import PlanetoidDataset
from rllm.nn.models import RECT_L
from rllm.transforms.graph_transforms import RECTTransform
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

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data")
data = PlanetoidDataset(path, args.dataset, force_reload=True)[0]

# Transform data
transform = RECTTransform()
data = transform(data)
zs_data = RemoveTrainingClasses(args.unseen_classes)(deepcopy(data)).to(device)


# Set up model, optimizer and loss function
model = RECT_L(
    in_dim=200,
    hidden_dim=200,
    dropout=args.dropout,
).to(device)
zs_data.y = model.get_semantic_labels(
    x=zs_data.x,
    y=zs_data.y,
    mask=zs_data.train_mask,
)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.wd,
)
loss_fn = torch.nn.MSELoss(reduction="sum")


def train():
    model.train()
    optimizer.zero_grad()
    out = model(zs_data.x, zs_data.adj)
    loss = loss_fn(out[zs_data.train_mask], zs_data.y)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test():
    model.eval()
    out = model.embed(zs_data.x, zs_data.adj).cpu()
    reg = LogisticRegression()
    reg.fit(out[data.train_mask].numpy(), data.y[data.train_mask].numpy())
    train_acc = reg.score(out[data.train_mask].numpy(), data.y[data.train_mask].numpy())
    test_acc = reg.score(out[data.test_mask].numpy(), data.y[data.test_mask].numpy())
    return train_acc, test_acc


metric = "Acc"
best_test_acc = 0
times = []
for epoch in range(1, args.epochs + 1):
    start = time.time()

    train_loss = train()
    train_acc, test_acc = test()

    if test_acc > best_test_acc:
        best_test_acc = test_acc

    times.append(time.time() - start)
    print(
        f"Epoch: [{epoch}/{args.epochs}] "
        f"Train Loss: {train_loss:.4f} Train {metric}: {train_acc:.4f} "
        f"Test Acc: {test_acc:.4f} "
    )

print(f"Mean time per epoch: {torch.tensor(times).mean():.4f}s")
print(f"Total time: {sum(times):.4f}s")
print(f"Best test acc: {best_test_acc:.4f}")
