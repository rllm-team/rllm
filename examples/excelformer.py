# The ExcelFormer method from the
# "ExcelFormer: A neural network surpassing GBDTs on tabular data" paper.
# ArXiv: https://arxiv.org/abs/2301.02819

# Datasets      Titanic    Jannis
# Metrics       AUC        Acc
# Rept.         -          0.735
# Ours          0.895      0.713
# Time          7.3s       251.1s

import argparse
import sys
import time
from typing import Any, Dict, List
import os.path as osp

from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn.functional as F

sys.path.append("./")
sys.path.append("../")
from rllm.types import ColType
from rllm.datasets import Jannis, Titanic
from rllm.transforms.table_transforms import DefaultTableTransform
from rllm.nn.conv.table_conv import ExcelFormerConv
from rllm.utils import data_aug

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", type=str, default="titanic", choices=["titanic", "jannis"]
)
parser.add_argument("--emb_dim", help="embedding dim", type=int, default=64)
parser.add_argument("--num_layers", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--wd", type=float, default=5e-4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--mix_type",
    type=str,
    default="none",
    choices=["none", "feat_mix", "hidden_mix", "niave_mix"],
)
parser.add_argument("--beta", type=float, default=0.5, help="Beta parameter for mixup")
parser.add_argument("--warm_up", type=int, default=0, help="Number of warmup epochs")
parser.add_argument(
    "--early_stop", type=int, default=20, help="Early stopping patience"
)
args = parser.parse_args()

# Set random seed and device
torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data")
if args.dataset.lower() == "jannis":
    data = Jannis(cached_dir=path)[0]
    metric = "acc"
else:
    data = Titanic(cached_dir=path)[0]
    metric = "auc"

# Transform data
transform = DefaultTableTransform(out_dim=args.emb_dim)
data = transform(data).to(device)
data.shuffle()

# Split dataset, here the ratio of train-val-test is 80%-10%-10%
train_loader, val_loader, test_loader = data.get_dataloader(
    train_split=0.8, val_split=0.1, test_split=0.1, batch_size=args.batch_size
)


# Define model
class ExcelFormer(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        metadata: Dict[ColType, List[Dict[str, Any]]],
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            ExcelFormerConv(
                conv_dim=hidden_dim,
                use_pre_encoder=True,
                metadata=metadata,
            )
        )
        for _ in range(num_layers - 1):
            self.convs.append(ExcelFormerConv(conv_dim=hidden_dim))

        self.fc = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x, mixup=False):
        # Apply mixup at feature level if needed
        feat_masks = None
        shuffled_ids = None

        # Get device from x (handle both dict and tensor)
        x_device = x[list(x.keys())[0]].device if isinstance(x, dict) else x.device

        if mixup and args.mix_type == "niave_mix":
            # Naive mixup at input level
            x, feat_masks, shuffled_ids = data_aug.mixup_data(x, beta=args.beta)
            if not isinstance(feat_masks, torch.Tensor):
                feat_masks = torch.tensor(feat_masks).to(x_device)
            shuffled_ids = torch.tensor(shuffled_ids).to(x_device)
        elif mixup and args.mix_type == "feat_mix":
            # Feature-level mixup
            x, feat_masks, shuffled_ids = data_aug.batch_feat_shuffle(x, beta=args.beta)
            shuffled_ids = torch.tensor(shuffled_ids).to(x_device)

        for conv in self.convs:
            x = conv(x)

        # Hidden-level mixup after convolutions
        if mixup and args.mix_type == "hidden_mix":
            x, feat_masks, shuffled_ids = data_aug.batch_dim_shuffle(x, beta=args.beta)
            shuffled_ids = torch.tensor(shuffled_ids).to(x_device)

        # Handle mean operation for both dict and tensor
        if isinstance(x, dict):
            # Concatenate all tensors in dict along feature dimension, then take mean
            x_concat = torch.cat([x[key] for key in sorted(x.keys())], dim=1)
            out = self.fc(x_concat.mean(dim=1))
        else:
            out = self.fc(x.mean(dim=1))

        if mixup and args.mix_type != "none":
            return out, feat_masks, shuffled_ids
        return out


# Set up model and optimizer
model = ExcelFormer(
    hidden_dim=args.emb_dim,
    out_dim=data.num_classes,
    num_layers=args.num_layers,
    metadata=data.metadata,
).to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.wd,
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)


def train(epoch: int) -> float:
    model.train()
    loss_accum = total_count = 0

    # Warmup learning rate
    if args.warm_up > 0 and epoch <= args.warm_up:
        lr = args.lr * epoch / args.warm_up
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    for batch in tqdm(train_loader, desc=f"Epoch: {epoch}"):
        x, y = batch
        optimizer.zero_grad()

        if args.mix_type == "none":
            # No mixup
            pred = model.forward(x, mixup=False)
            loss = F.cross_entropy(pred, y.long())
        else:
            # Apply mixup
            pred, feat_masks, shuffled_ids = model.forward(x, mixup=True)

            # Calculate lambdas based on mixup type
            if args.mix_type == "feat_mix":
                # Convert boolean mask to float and average across features
                lambdas = feat_masks.float().mean(dim=1)  # Average across features
                lambdas2 = 1 - lambdas
            else:  # hidden_mix or niave_mix
                lambdas = (
                    feat_masks
                    if isinstance(feat_masks, torch.Tensor)
                    else torch.tensor(feat_masks).to(pred.device)
                )
                lambdas2 = 1 - lambdas

            # Compute mixup loss
            if data.num_classes == 2:  # Binary classification
                loss = lambdas * F.cross_entropy(
                    pred, y.long(), reduction="none"
                ) + lambdas2 * F.cross_entropy(
                    pred, y[shuffled_ids].long(), reduction="none"
                )
                loss = loss.mean()
            else:  # Multi-class classification
                loss = lambdas * F.cross_entropy(
                    pred, y.long(), reduction="none"
                ) + lambdas2 * F.cross_entropy(
                    pred, y[shuffled_ids].long(), reduction="none"
                )
                loss = loss.mean()

        loss.backward()
        optimizer.step()
        loss_accum += float(loss) * y.size(0)
        total_count += y.size(0)

    return loss_accum / total_count


@torch.no_grad()
def test(loader: DataLoader, model, metric: str = "auc") -> float:
    model.eval()
    all_preds = []
    all_labels = []

    for x, y in loader:
        pred = model(x, mixup=False)
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
best_test_metric = 0
no_improvement = 0
times = []

for epoch in range(1, args.epochs + 1):
    start = time.time()

    train_loss = train(epoch)
    train_metric = test(train_loader, model, metric)
    val_metric = test(val_loader, model, metric)
    tmp_test_metric = test(test_loader, model, metric)

    if val_metric > best_val_metric:
        best_val_metric = val_metric
        test_metric = tmp_test_metric
        no_improvement = 0
        print(
            f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f}, Train {metric}: {train_metric:.4f}, "
            f"Val {metric}: {val_metric:.4f}, Test {metric}: {tmp_test_metric:.4f} <<< BEST VALIDATION EPOCH"
        )
    else:
        no_improvement += 1
        print(
            f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f}, Train {metric}: {train_metric:.4f}, "
            f"Val {metric}: {val_metric:.4f}, Test {metric}: {tmp_test_metric:.4f}"
        )

    if tmp_test_metric > best_test_metric:
        best_test_metric = tmp_test_metric

    times.append(time.time() - start)

    # Learning rate scheduling (after warmup)
    if args.warm_up == 0 or epoch > args.warm_up:
        scheduler.step()

    # Early stopping
    if no_improvement >= args.early_stop:
        print(f"Early stopping triggered after {epoch} epochs")
        break

print(f"\nMean time per epoch: {torch.tensor(times).mean():.4f}s")
print(f"Total time: {sum(times):.4f}s")
print(f"Best Val {metric}: {best_val_metric:.4f}")
print(f"Test {metric} at best Val: {test_metric:.4f}")
print(f"Best Test {metric}: {best_test_metric:.4f}")
