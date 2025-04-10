from typing import Optional, List, Tuple, Callable

import torch
from torch import Tensor

from rllm.data import GraphData


class NeighborLoader(torch.utils.data.DataLoader):
    r"""The neighbor sampler from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper, which allows
    for mini-batch training of GNNs on large-scale graphs where full-batch
    training is not feasible.

    Args:
        data (GraphData): The graph data to be sampled.
        num_neighbors (List[int]): The number of neighbors to sample
            for each node in each layer.
        seeds (Optional[Tensor]): The nodes to sample from. If None,
            all nodes will be used.
        transform (Optional[Callable]): A function/transform that takes
            in a graph and returns a transformed version. The data
            loader will use this function to transform the graph before
            returning it.
        replace (bool, optional): Whether to sample with replacement.
            Default is False.
        shuffle (bool, optional): Whether to shuffle the data at every
            epoch. Default is False.
        batch_size (int, optional): How many samples per batch to load.
            Default is 1.
        num_workers (int, optional): How many subprocesses to use for
            data loading. Default is 0.
        **kwargs: Additional keyword arguments to be passed to the
            `torch.utils.data.DataLoader` class.
    """
    def __init__(
        self,
        data: GraphData,
        num_neighbors: List[int],
        seeds: Optional[Tensor] = None,
        transform: Optional[Callable] = None,
        replace: bool = False,
        shuffle: bool = False,
        batch_size: int = 1,
        num_workers: int = 0,
        **kwargs,
    ):
        kwargs.pop("dataset", None)
        kwargs.pop("collate_fn", None)

        self.device = data.device
        self.num_neighbors = num_neighbors
        self.replace = replace
        self.transform = transform

        self.num_nodes = data.num_nodes

        if seeds is None:
            seeds = torch.arange(self.num_nodes, dtype=torch.long)
        elif not isinstance(seeds, Tensor):
            seeds = torch.tensor(seeds, dtype=torch.long, device=self.device)
        elif seeds.dtype == torch.bool:
            seeds = seeds.nonzero(as_tuple=False).flatten()

        # prepare csc for sampling
        self._build_csc(data)

        super().__init__(
            dataset=seeds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            **kwargs,
        )

    def _build_csc(self, data: GraphData):
        r"""Build a compressed sparse column (CSC) representation of the graph
        for efficient neighbor sampling.
        """
        if hasattr(data, "edge_index"):
            edge_index = data.edge_index
        elif hasattr(data, "adj"):
            edge_index = data.adj.coalesce().indices()
        order = torch.argsort(edge_index[1])
        self.dst_sorted = edge_index[1][order]
        self.src_sorted = edge_index[0][order]
        cnts = torch.bincount(self.dst_sorted, minlength=self.num_nodes)
        self.col_ptr = torch.empty(
            self.num_nodes + 1, dtype=torch.long, device=self.device
        )
        self.col_ptr[0] = 0
        self.col_ptr[1:] = torch.cumsum(cnts, dim=0)

    def get_in_neighbors(self, node: int) -> torch.Tensor:
        r"""Get the in-neighbors of a given node in the graph.

        Args:
            node (int): The node for which to get the in-neighbors.
        """
        start = self.col_ptr[node].item()
        end = self.col_ptr[node + 1].item()
        return self.src_sorted[start:end]

    def sample_neighbors_one_layer(
        self, seed_nodes: List[int], num_neighbor: int
    ) -> Tuple[Tensor, Tensor]:
        r"""Sample neighbors for a given set of seed nodes.

        Args:
            seed_nodes (List[int]): The nodes to sample neighbors from.
            num_neighbor (int): The number of neighbors to sample for
                each node.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the sampled source
            nodes and destination nodes.
        """
        sampled_src_list = []
        dst_list = []
        for i, node in enumerate(seed_nodes):
            neighbors = self.get_in_neighbors(node)
            n_neighbors = neighbors.numel()
            if n_neighbors == 0:
                continue
            elif num_neighbor < 0 or n_neighbors < num_neighbor:
                sampled = neighbors
            else:
                perm = torch.randperm(n_neighbors, device=self.device)[:num_neighbor]
                sampled = neighbors[perm]

            sampled_src_list.append(sampled)
            dst_list.append(
                torch.full(
                    (sampled.numel(),), node, dtype=torch.long, device=self.device
                )
            )
        if sampled_src_list:
            return torch.cat(sampled_src_list), torch.cat(dst_list)
        else:
            return (
                torch.empty((0,), dtype=torch.long, device=self.device),
                torch.empty((0,), dtype=torch.long, device=self.device),
            )

    def collate_fn(
        self,
        batch: List[Tensor],
    ) -> Tuple[int, Tensor, List[Tensor]]:
        r"""Collate function for the NeighborLoader. This function
        is responsible for sampling neighbors for each node in the
        batch and returning the sampled nodes and their corresponding
        adjacency lists.
        """
        batch = torch.tensor(batch, dtype=torch.long).tolist()
        raw_adjs = []
        seed_nodes = batch
        n_id = batch.copy()
        seen = set(n_id)
        for num_neighbor in self.num_neighbors:
            sampled_src, dst = self.sample_neighbors_one_layer(seed_nodes, num_neighbor)
            raw_adjs.append((sampled_src, dst))

            for node in sampled_src.tolist():
                if node not in seen:
                    seen.add(node)
                    n_id.append(node)
            seed_nodes = sampled_src.unique().tolist()

        n_id = torch.tensor(n_id, dtype=torch.long, device=self.device)
        sorted_, perm = torch.sort(n_id)

        adjs = []
        for i, (src, dst) in enumerate(raw_adjs):
            if src.numel() == 0:
                adjs.append(torch.empty((0,), dtype=torch.long, device=self.device))
                continue
            src_ = perm[torch.searchsorted(sorted_, src)]
            dst_ = perm[torch.searchsorted(sorted_, dst)]
            edge_index = torch.stack([src_, dst_], dim=0)
            size = edge_index.max() + 1
            adj = torch.sparse_coo_tensor(
                indices=edge_index,
                values=torch.ones(edge_index.shape[1], device=self.device),
                size=(size, size),
                device=self.device,
            )
            if self.transform is not None:
                adj = self.transform(adj)
            adjs.append(adj)
        return len(batch), n_id, adjs
