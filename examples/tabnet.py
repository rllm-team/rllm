import argparse
import os.path as osp
import sys
from typing import Any, Dict, List

from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append("./")
sys.path.append("../")
from rllm.types import ColType
from rllm.datasets.titanic import Titanic
from rllm.transforms.table_transforms import TabNetTransform
from rllm.nn.models import TabNet

parser = argparse.ArgumentParser()
parser.add_argument("--dim", help="embedding dim", type=int, default=32)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--wd", type=float, default=5e-4)
args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare datasets
path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data")
dataset = Titanic(cached_dir=path)[0]
dataset.to(device)
dataset.shuffle()

# Split dataset, here the ratio of train-val-test is 80%-10%-10%
train_loader, val_loader, test_loader = dataset.get_dataloader(
    0.8, 0.1, 0.1, batch_size=args.batch_size
)


# Set up model and optimizer
class TabNetModel(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        col_stats_dict: Dict[ColType, List[Dict[str, Any]]],
    ):
        super().__init__()
        self.transform = TabNetTransform(
            out_dim=hidden_dim,
            col_stats_dict=col_stats_dict,
        )
        self.transform.post_init()
        self.backbone = TabNet(
            out_dim=out_dim,  # dataset.num_classes,
            cat_emb_dim=hidden_dim,  # args.dim,
            num_emb_dim=hidden_dim,  # args.dim,
            col_stats_dict=dataset.stats_dict,
        )

    def forward(self, x):
        x = self.transform(x)
        out = self.backbone(x.reshape(x.size(0), -1))
        return out


model = TabNetModel(
    out_dim=dataset.num_classes,
    hidden_dim=args.dim,
    col_stats_dict=dataset.stats_dict,
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.wd,
)


def train(epoch: int, lambda_sparse: float = 1e-4) -> float:
    model.train()
    loss_accum = total_count = 0
    for batch in tqdm(train_loader, desc=f"Epoch: {epoch}"):
        x, y = batch
        pred, M_loss = model.forward(x)
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
        x, y = batch
        pred, _ = model.forward(x)
        all_labels.append(y.cpu())
        all_preds.append(pred[:, 1].detach().cpu())
    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    # if np.isnan(all_labels).any():
    #     print("NaN found in all_labels")
    # if np.isnan(all_preds).any():
    #     print("NaN found in all_preds")
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
