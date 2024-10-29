import argparse
import os.path as osp
import sys

sys.path.append("../")

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from rllm.types import ColType
from rllm.datasets.titanic import Titanic
from rllm.transforms.table_transforms import TabNetTransform
from rllm.nn.models import TabNet

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    default="titanic",
    choices=[
        "titanic",
    ],
)
parser.add_argument("--dim", help="embedding dim", type=int, default=32)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--compile", action="store_true")
args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare datasets
path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data")
dataset = Titanic(cached_dir=path)[0]
dataset.to(device)
dataset.shuffle()

# Split dataset, here the ratio of train-val-test is 20%-40%-40%
train_dataset, val_dataset, test_dataset = dataset.get_dataset(0.8, 0.1, 0.1)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)


# Set up model and optimizer
class TabNetModel(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        col_stats_dict: dict[ColType, list[dict[str,]]],
    ):
        super().__init__()
        self.transform = TabNetTransform(
            out_dim=hidden_dim,
            col_stats_dict=col_stats_dict,
        )
        self.backbone = TabNet(
            output_dim=output_dim,  # dataset.num_classes,
            cat_emb_dim=hidden_dim,  # args.dim,
            col_stats_dict=dataset.stats_dict,
        )

    def forward(self, x):
        x, _ = self.transform(x)
        out = self.backbone(x)
        return out


model = TabNetModel(
    output_dim=dataset.num_classes,
    hidden_dim=args.dim,
    col_stats_dict=dataset.stats_dict,
).to(device)

model = torch.compile(model, dynamic=True) if args.compile else model
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def train(epoch: int, lambda_sparse: float = 1e-4) -> float:
    model.train()
    loss_accum = total_count = 0
    for batch in tqdm(train_loader, desc=f"Epoch: {epoch}"):
        feat_dict, y = batch
        pred, M_loss = model.forward(feat_dict)
        loss = F.cross_entropy(pred, y.long())
        loss = loss - lambda_sparse * M_loss
        optimizer.zero_grad()
        loss.backward()
        loss_accum += float(loss) * y.size(0)
        total_count += y.size(0)
        optimizer.step()
    return loss_accum / total_count


@torch.no_grad()
def test(loader: DataLoader) -> float:
    model.eval()
    all_preds = []
    all_labels = []
    for batch in loader:
        feat_dict, y = batch
        pred, M_loss = model.forward(feat_dict)
        all_labels.append(y.cpu())
        all_preds.append(pred[:, 1].detach().cpu())
    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()

    # Compute the overall AUC
    overall_auc = roc_auc_score(all_labels, all_preds)
    return overall_auc


metric = "AUC"
best_val_metric = 0
best_test_metric = 0
for epoch in range(1, args.epochs + 1):
    train_loss = train(epoch)
    train_metric = test(train_loader)
    val_metric = test(val_loader)
    test_metric = test(test_loader)

    if val_metric > best_val_metric:
        best_val_metric = val_metric
        best_test_metric = test_metric

    print(
        f"Train Loss: {train_loss:.4f}, Train {metric}: {train_metric:.4f}, "
        f"Val {metric}: {val_metric:.4f}, Test {metric}: {test_metric:.4f}"
    )
    optimizer.step()

print(
    f"Best Val {metric}: {best_val_metric:.4f}, "
    f"Best Test {metric}: {best_test_metric:.4f}"
)
