from typing import Dict

import torch
from torch import Tensor

from rllm.data import HeteroGraphData
from rllm.transforms.utils import BaseTransform
from rllm.transforms.utils.functional.metapath_prop import metapath_propagate


class MetaPathProp(BaseTransform):
    r"""Generate multi-hop meta-path features for a target node type from a
    heterogeneous mini-batch, following MetaRTL
    `"MetaRTL: Meta-path Attention Enhanced Relational Table Learning"`.

    This is the pre-processing step between the stage-one encoder and the
    stage-two fusion model of MetaRTL. Given pre-computed node embeddings
    :obj:`x_dict` (e.g. produced by a frozen encoder), it propagates features
    along each edge type for multiple hops using a row-normalized weighted
    mean, and stacks the resulting meta-path features into a single tensor of
    shape :obj:`[B, M, D]`, where :obj:`M` is the number of meta-paths and
    :obj:`B` the number of target seed nodes.

    Note:
        Unlike node- or edge-wise graph transforms, this operation consumes
        both the graph structure and externally provided node features, and
        returns a new feature tensor rather than a mutated graph. It therefore
        lives under :mod:`rllm.transforms.utils` rather than the node/edge
        graph transforms.

    Args:
        target_node_type (str): The node type to collect meta-path features for.
        min_hops (int): Minimum hop count to keep. (default: :obj:`0`)
        max_hops (int): Maximum number of propagation hops. (default: :obj:`3`)

    Example:
        >>> prop = MetaPathProp(target_node_type="drivers", max_hops=3)
        >>> x = prop(batch, x_dict)  # (B, M, D)
    """

    def __init__(
        self,
        target_node_type: str,
        min_hops: int = 0,
        max_hops: int = 3,
    ) -> None:
        self.target_node_type = target_node_type
        self.min_hops = min_hops
        self.max_hops = max_hops

    def __call__(
        self,
        data: HeteroGraphData,
        x_dict: Dict[str, Tensor],
    ) -> Tensor:
        # Meta-path propagation reads from the graph without mutating it, so we
        # forward the data directly instead of shallow-copying as in the base
        # class. The second ``x_dict`` argument carries the node features.
        return self.forward(data, x_dict)

    def forward(
        self,
        data: HeteroGraphData,
        x_dict: Dict[str, Tensor],
    ) -> Tensor:
        r"""Run meta-path propagation on the mini-batch.

        Args:
            data (HeteroGraphData): Batched heterogeneous graph providing
                :obj:`edge_index_dict` and per-node-type node counts.
            x_dict (Dict[str, Tensor]): Base node features by node type.

        Returns:
            Tensor: Meta-path features of shape :obj:`[B, M, D]` ordered by
            meta-path name.
        """
        edge_index_dict = {
            edge_type: edge_index
            for edge_type, edge_index in data.edge_index_dict.items()
            if edge_index.numel() > 0
        }
        num_nodes_dict = {
            node_type: node_store.num_nodes
            for node_type, node_store in data.node_items()
        }

        feats = metapath_propagate(
            edge_index_dict=edge_index_dict,
            x_dict=x_dict,
            num_nodes_dict=num_nodes_dict,
            target_node_type=self.target_node_type,
            min_hops=self.min_hops,
            max_hops=self.max_hops,
        )
        ordered_names = sorted(feats.keys())
        return torch.cat([feats[k] for k in ordered_names], dim=1)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"target_node_type={self.target_node_type}, "
            f"min_hops={self.min_hops}, max_hops={self.max_hops})"
        )
