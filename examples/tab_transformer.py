# The TabTransformer method from the
# "TabTransformer: Tabular Data Modeling Using Contextual Embeddings" paper.
# ArXiv: https://arxiv.org/abs/2012.06678

# Datasets      Titanic    Adult
# Metrics       Acc        AUC
# Rept.         -          0.737
# Ours          0.842      0.850
# Time          5.26s      251.1s

import argparse
import sys
import time
from typing import Any, Dict, List
import os.path as osp

from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

sys.path.append("./")
sys.path.append("../")
from rllm.types import ColType
from rllm.datasets import Titanic, Adult
from rllm.transforms.table_transforms import TabTransformerTransform
from rllm.nn.conv.table_conv import TabTransformerConv

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", type=str, default="titanic", choices=["titanic", "adult"]
)
parser.add_argument("--emb_dim", help="embedding dim", type=int, default=32)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--num_heads", type=int, default=8)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--wd", type=float, default=5e-4)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

# Set random seed and device
torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data")
if args.dataset.lower() == "adult":
    data = Adult(cached_dir=path)[0]
    metric = "auc"
else:
    data = Titanic(cached_dir=path)[0]
    metric = "acc"

# Transform data
transform = TabTransformerTransform(out_dim=args.emb_dim)
data = transform(data).to(device)
data.shuffle()

# Split dataset, here the ratio of train-val-test is 80%-10%-10%
train_loader, val_loader, test_loader = data.get_dataloader(
    train_split=0.8, val_split=0.1, test_split=0.1, batch_size=args.batch_size
)


# Define model
class TabTransformer(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        num_heads: int,
        metadata: Dict[ColType, List[Dict[str, Any]]],
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            TabTransformerConv(
                conv_dim=hidden_dim,
                num_heads=num_heads,
                use_pre_encoder=True,
                metadata=metadata,
            )
        )
        for _ in range(num_layers - 1):
            self.convs.append(
                TabTransformerConv(conv_dim=hidden_dim, num_heads=num_heads)
            )

        self.fc = torch.nn.Linear(
            len(metadata[ColType.CATEGORICAL]) * hidden_dim
            + len(metadata[ColType.NUMERICAL]),
            out_dim,
        )

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        x[ColType.CATEGORICAL] = x[ColType.CATEGORICAL].flatten(1)
        x[ColType.NUMERICAL] = x[ColType.NUMERICAL].flatten(1)
        x = torch.cat(list(x.values()), dim=1)
        out = self.fc(x)
        return out


# Set up model and optimizer
model = TabTransformer(
    hidden_dim=args.emb_dim,
    out_dim=data.num_classes,
    num_layers=args.num_layers,
    num_heads=args.num_heads,
    metadata=data.metadata,
).to(device)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.wd,
)


def train(epoch: int) -> float:
    model.train()
    loss_accum = total_count = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch: {epoch}"):
        x, y = batch
        pred = model.forward(x)
        loss = F.cross_entropy(pred, y.long())
        optimizer.zero_grad()
        loss.backward()
        loss_accum += float(loss) * y.size(0)
        total_count += y.size(0)
        optimizer.step()
    return loss_accum / total_count


@torch.no_grad()
def test(loader: DataLoader, metric: str = "auc") -> float:
    model.eval()
    all_preds = []
    all_labels = []

    for x, y in loader:
        pred = model(x)
        probs = torch.softmax(pred, dim=1)
        all_labels.append(y.cpu())
        all_preds.append(probs.detach().cpu())
    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_preds, dim=0).numpy()
    num_classes = len(torch.unique(torch.tensor(all_labels)))

    if metric.lower() == "auc":
        if num_classes == 2:
            score = float(roc_auc_score(all_labels, all_probs[:, 1]))
        else:
            score = float(roc_auc_score(all_labels, all_probs, multi_class="ovr"))
    elif metric.lower() == "acc":
        preds = torch.argmax(torch.tensor(all_probs), dim=1).numpy()
        score = float(accuracy_score(all_labels, preds))
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    return score


best_val_metric = test_metric = 0
times = []
for epoch in range(1, args.epochs + 1):
    start = time.time()

    train_loss = train(epoch)
    train_metric = test(train_loader, metric)
    val_metric = test(val_loader, metric)
    tmp_test_metric = test(test_loader, metric)

    if val_metric > best_val_metric:
        best_val_metric = val_metric
        test_metric = tmp_test_metric

    times.append(time.time() - start)
    print(
        f"Epoch: [{epoch}/{args.epochs}]"
        f"Train Loss: {train_loss:.4f}, Train {metric}: {train_metric:.4f}, "
        f"Val {metric}: {val_metric:.4f}, Test {metric}: {tmp_test_metric:.4f}"
    )

print(f"Mean time per epoch: {torch.tensor(times).mean():.4f}s")
print(f"Total time: {sum(times):.4f}s")
print(f"Test {metric} at best Val: {test_metric:.4f}")
