from typing import Optional, List

import pandas as pd
import numpy as np
import torch
from torch import Tensor

from rllm.data import GraphData
from rllm.transforms.graph_transforms import GCNTransform
from rllm.transforms.table_transforms import TabTransformerTransform


def reorder_ids(
    relation_df: pd.DataFrame,
    src_col_name: str,
    tgt_col_name: str,
    n_src: int,
):
    r"""Reorders the IDs in the relationship DataFrame by adjusting the
    original source IDs and target column IDs.

    Args:
        relation_df (pd.DataFrame): DataFrame containing the relationships.
        src_col_name (str): Name of the source column in the DataFrame.
        tgt_col_name (str): Name of the target column in the DataFrame.
        n_src (int): Number of source nodes.
    """
    # Making relationship
    ordered_rating = relation_df.assign(
        **{
            src_col_name: relation_df[src_col_name] - 1,
            tgt_col_name: relation_df[tgt_col_name] + n_src - 1,
        }
    )

    return ordered_rating


def build_homo_graph(
    relation_df: pd.DataFrame,
    n_all: int,
    x: Optional[Tensor] = None,
    y: Optional[Tensor] = None,
    edge_per_node: Optional[int] = None,
):
    r"""Use the given dataframe to construct a simple undirected and
        unweighted graph with only two types of nodes and one type of edge.

    Args:
        relation_df (pd.DataFrame): The given dataframe, where the first two
            columns represent two types of nodes that may be connected in a
            homogeneous graph (abbreviated as sorce nodes and target nodes).
            Assume that the indices of source and target nodes
            in the dataframe start from 0.
        n_all (int): Total amount of nodes.
        x (Tensor): Features of nodes.
        y (Optional[Tensor]): Labels of (part) nodes.
        edge_per_node (Optional[int]):
            specifying the maximum number of edges to keep for each node.
    """

    # Get adj
    src_nodes, tgt_nodes = torch.from_numpy(relation_df.iloc[:, :2].values).t()
    indices = torch.cat(
        [
            torch.stack([src_nodes, tgt_nodes], dim=0),  # src -> tgt
            torch.stack([tgt_nodes, src_nodes], dim=0),  # tgt -> src
        ],
        dim=1,
    )

    if edge_per_node is not None:
        unique_nodes = torch.unique(indices)
        mask = torch.zeros(indices.shape[1], dtype=torch.bool)

        for node in unique_nodes:
            # Find neighbors
            node_mask = (indices[0] == node) | (indices[1] == node)
            node_edges = torch.nonzero(node_mask).squeeze()

            # Randomly select `edge_per_node` edges
            if node_edges.numel() > edge_per_node:
                selected_edges = node_edges[
                    torch.randperm(node_edges.numel())[:edge_per_node]
                ]
                mask[selected_edges] = True
            else:
                mask[node_edges] = True

        indices = indices[:, mask]

    values = torch.ones((indices.shape[1],), dtype=torch.float32)
    adj = torch.sparse_coo_tensor(indices, values, (n_all, n_all))

    # Construct graph
    graph = GraphData(x=x, y=y, adj=adj)

    return graph


def build_batch_homo_graph(blocks, target_table):
    r"""Like as build_homo_graph(), only build a simple undirected,
    and unweighted edge list here.

    edge_table.fkey1 ----> node_table.pkey

    edge_table.fkey2 ----> node_table.pkey
    """
    assert len(blocks) == 2
    edge_list = [[], []]
    oind: List[int] = target_table.oind
    n_nodes = len(oind)

    for fkey_id, pkey_id_1 in zip(blocks[0].edge_list[0], blocks[0].edge_list[1]):
        fkey_id = np.where(blocks[1].edge_list[0] == fkey_id)[0]
        pkey_id_2 = blocks[1].edge_list[1][fkey_id]

        pkey_id_2 = pkey_id_2[0]

        # transfer oind -> new id
        pkey_id_1 = oind.index(pkey_id_1)
        pkey_id_2 = oind.index(pkey_id_2)

        # add undirected edge
        edge_list[0].extend([pkey_id_1, pkey_id_2])
        edge_list[1].extend([pkey_id_2, pkey_id_1])

    edge_list = torch.tensor(edge_list, dtype=torch.long)
    values = torch.ones((edge_list.shape[1],), dtype=torch.float32)
    adj = torch.sparse_coo_tensor(edge_list, values, (n_nodes, n_nodes))

    # Construct graph
    graph = GraphData(adj=adj)

    return graph


def data_prepare(dataset, dataset_name, device):
    if dataset_name == "tlf2k":
        # Get the required data
        artist_table, ua_table, _ = dataset.data_list
        emb_size = 384  # Use the same embedding size as BERT for simplicity
        artist_size = len(artist_table)
        user_size = ua_table.df["userID"].max()

        target_table = artist_table.to(device)
        non_table_embeddings = torch.randn((user_size, emb_size)).to(device)

        ordered_ua = reorder_ids(
            relation_df=ua_table.df,
            src_col_name="artistID",
            tgt_col_name="userID",
            n_src=artist_size,
        )

        # Build graph
        graph = build_homo_graph(
            relation_df=ordered_ua,
            n_all=artist_size + user_size,
        ).to(device)
    elif dataset_name == "tml1m":
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
        non_table_embeddings = movie_embeddings.to(device)

        # Build graph
        graph = build_homo_graph(
            relation_df=ordered_rating,
            n_all=user_size + movie_embeddings.size(0),
        ).to(device)
    elif dataset_name == "tacm12k":
        # Get the required data
        (
            papers_table,
            _,
            citations_table,
            _,
            _,
            _,
        ) = dataset.data_list
        emb_size = 384
        target_table = papers_table.to(device)
        non_table_embeddings = None

        # Build graph
        graph = build_homo_graph(
            relation_df=citations_table.df,
            n_all=len(papers_table),
        ).to(device)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Transform data
    table_transform = TabTransformerTransform(
        out_dim=emb_size, metadata=target_table.metadata
    )
    target_table = table_transform(data=target_table)
    graph_transform = GCNTransform()
    adj = graph_transform(data=graph).adj
    target_table.y = target_table.y.long().to(device)

    return target_table, non_table_embeddings, adj, emb_size
