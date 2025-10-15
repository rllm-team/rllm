import time
import argparse
import sys
import os.path as osp

import torch
import torch.nn.functional as F

sys.path.append("./")
sys.path.append("../")
from rllm.datasets import TLF2KDataset, TACM12KDataset, TML1MDataset
from rllm.transforms.graph_transforms import GCNTransform
from rllm.transforms.table_transforms import TabTransformerTransform
from rllm.nn.conv.graph_conv import GCNConv
from rllm.nn.conv.table_conv import TabTransformerConv
from rllm.nn.models import BRIDGE, TableEncoder, GraphEncoder
from examples.bridge.utils import build_homo_graph, reorder_ids


def train(model, optimizer, target_table, non_table_embeddings, adj, y, train_mask):
    model.train()
    optimizer.zero_grad()
    logits = model(table=target_table, non_table=non_table_embeddings, adj=adj)
    loss = F.cross_entropy(logits[train_mask].squeeze(), y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(model, target_table, non_table_embeddings, adj, y, masks):
    model.eval()
    logits = model(table=target_table, non_table=non_table_embeddings, adj=adj)
    preds = logits.argmax(dim=1)
    accs = []
    for mask in masks:
        correct = float(preds[mask].eq(y[mask]).sum().item())
        accs.append(correct / int(mask.sum()))
    return accs


def data_prepare(dataset, dataset_name, device):
    if dataset_name == "TLF2K" or dataset_name == "tlf2k":
        artist_table, ua_table, _ = dataset.data_list
        emb_size = 384
        artist_size = len(artist_table)
        user_size = ua_table.df["userID"].max()

        target_table = artist_table.to(device)
        y = artist_table.y.long().to(device)
        num_classes = artist_table.num_classes
        non_table_embeddings = torch.randn((user_size, emb_size)).to(device)

        ordered_ua = reorder_ids(
            relation_df=ua_table.df,
            src_col_name="artistID",
            tgt_col_name="userID",
            n_src=artist_size,
        )

        graph = build_homo_graph(
            relation_df=ordered_ua,
            n_all=artist_size + user_size,
        ).to(device)

        train_mask, val_mask, test_mask = (
            artist_table.train_mask,
            artist_table.val_mask,
            artist_table.test_mask,
        )

    elif dataset_name == "TML1M" or dataset_name == "tml1m":

        # Get the required data
        (
            user_table,
            _,
            rating_table,
            movie_embeddings,
        ) = dataset.data_list
        emb_size = movie_embeddings.size(1)
        user_size = len(user_table)

        ordered_rating = reorder_ids(
            relation_df=rating_table.df,
            src_col_name="UserID",
            tgt_col_name="MovieID",
            n_src=user_size,
        )
        target_table = user_table.to(device)
        y = user_table.y.long().to(device)
        num_classes = user_table.num_classes
        non_table_embeddings = movie_embeddings.to(device)

        # Build graph
        graph = build_homo_graph(
            relation_df=ordered_rating,
            n_all=user_size + movie_embeddings.size(0),
        ).to(device)

        # Split data
        train_mask, val_mask, test_mask = (
            user_table.train_mask,
            user_table.val_mask,
            user_table.test_mask,
        )

    elif dataset_name == "TACM12K" or dataset_name == "tacm12k":
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
        num_classes = papers_table.num_classes
        non_table_embeddings = paper_embeddings.to(device)

        # Build graph
        graph = build_homo_graph(
            relation_df=citations_table.df,
            n_all=len(papers_table),
        ).to(device)

        # Split data
        train_mask, val_mask, test_mask = (
            target_table.train_mask,
            target_table.val_mask,
            target_table.test_mask,
        )

    # Transform data
    table_transform = TabTransformerTransform(
        out_dim=emb_size, metadata=target_table.metadata
    )
    target_table = table_transform(data=target_table)
    graph_transform = GCNTransform()
    adj = graph_transform(data=graph).adj

    return target_table, non_table_embeddings, adj, y, num_classes, emb_size, train_mask, val_mask, test_mask


def train_bridge_model(model, target_table, non_table_embeddings, adj, y, num_classes, emb_size, train_mask, val_mask, test_mask, epochs, lr, wd, device):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    metric = "Acc"
    best_val_acc = test_acc = 0
    times = []
    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss = train(model, optimizer, target_table, non_table_embeddings, adj, y, train_mask)
        train_acc, val_acc, tmp_test_acc = test(model, target_table, non_table_embeddings, adj, y, [train_mask, val_mask, test_mask])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc

        times.append(time.time() - start)
        print(
            f"Epoch: [{epoch}/{epochs}]"
            f"Train Loss: {train_loss:.4f} Train {metric}: {train_acc:.4f} "
            f"Val {metric}: {val_acc:.4f}, Test {metric}: {tmp_test_acc:.4f} "
        )

    print(f"Mean time per epoch: {torch.tensor(times).mean():.4f}s")
    print(f"Total time: {sum(times):.4f}s")
    print(f"Test {metric} at best Val: {test_acc:.4f}")

    return model, best_val_acc, test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="tlf2k", choices=["tlf2k", "tml1m", "tacm12k"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wd", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data")
    if args.dataset == "TLF2K" or args.dataset == "tlf2k":
        dataset = TLF2KDataset(cached_dir=path, force_reload=True)
    elif args.dataset == "TML1M" or args.dataset == "tml1m":
        dataset = TML1MDataset(cached_dir=path, force_reload=True)
    elif args.dataset == "TACM12K" or args.dataset == "tacm12k":
        dataset = TACM12KDataset(cached_dir=path, force_reload=True)

    target_table, non_table_embeddings, adj, y, num_classes, emb_size, train_mask, val_mask, test_mask = data_prepare(dataset, args.dataset, device)

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

    train_bridge_model(model, target_table, non_table_embeddings, adj, y, num_classes, emb_size, train_mask, val_mask, test_mask,  args.epochs, args.lr, args.wd, device)