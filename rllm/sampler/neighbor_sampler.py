from collections import defaultdict
from typing import List, Optional

import torch
from torch import Tensor
#from torch_sparse import SparseTensor

#from rllm.utils.sparse import is_torch_sparse_tensor


class NeighborSampler(torch.utils.data.DataLoader):
    r"""The neighbor sampler from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper, which allows
    for mini-batch training of GNNs on large-scale graphs where full-batch
    training is not feasible.

    Args:
        adj (Tensor):
            an adjacency matrix which defines the underlying graph
            connectivity flow.
        num_samples (List):
            The number of neighbors to sample for each node in each
            layer. If set to `num_samples[l] = -1`,
            all neighbors are included in layer l.
        node_idx (Tensor):
            The nodes that should be considered for creating mini-batches.
            If set to `None`, all nodes will be considered.
        **kwargs (Optional):
            Additional arguments of `torch.utils.data.Dataloader`,
            such as `batch_size`, `shuffle` and `num_workers`.
    """
    def __init__(self,
                 adj: Tensor,
                 num_samples: List[int],
                 node_idx: Optional[Tensor] = None,
                 **kwargs):

        assert is_torch_sparse_tensor(adj)

        adj = adj.coalesce()
        edge = adj.indices()
        self.adj = SparseTensor(row=edge[0],
                                col=edge[1],
                                value=adj.indices(),
                                sparse_sizes=adj.shape,
                                is_sorted=True)
        self.num_samples = num_samples

        if node_idx is None:
            node_idx = torch.arange(adj.shape[0])
        elif node_idx.dtype == torch.bool:
            node_idx = node_idx.nonzero().view(-1)

        super().__init__(
            node_idx.view(-1).tolist(),
            collate_fn=self.sample,
            **kwargs
        )

    def sample(self, batch):
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)

        neighs_samples = []
        n_id = batch
        for n in self.num_samples:
            neighs = defaultdict(list)
            subadj, n_id = self.adj.sample_adj(n_id, n, replace=False)
            row, col, _ = subadj.coo()
            for i, j in zip(row.tolist(), col.tolist()):
                neighs[n_id[i]].append(neighs[n_id[j]])
            neighs = {k: torch.as_tensor(v) for k, v in neighs.items()}
            neighs_samples.append(neighs)

        neighs_samples = neighs_samples[0] if len(neighs_samples) == 1\
            else neighs_samples[::-1]
        return (batch, neighs_samples)
