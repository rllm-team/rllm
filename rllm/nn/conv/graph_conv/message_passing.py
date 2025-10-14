from abc import ABC
import inspect
from collections import OrderedDict
from functools import lru_cache
from typing import Tuple, Callable, Dict, Any, Union, Optional, overload

import torch
from torch import Tensor
from torch.sparse import Tensor as SparseTensor

from rllm.nn.conv.graph_conv.aggrs import Aggregator


class MessagePassing(torch.nn.Module, ABC):
    r"""Base class for message passing.

    Message passing is the general framework for graph neural networks.
    Its forward formula is defined as:

    .. math::
        \mathbf{x}_i^{(k+1)} = \text{Update}^{(k)}
        \left( \mathbf{x}_i^{(k)},
        \text{Aggregate}^{(k)} \left( \left\{ \text{Message}^{(k)} \left(
        \mathbf{x}_i^{(k)}, \mathbf{x}_j^{(k)}, \mathbf{e}_{j,i}^{(k)}
        \right) \right\}_{j \in \mathcal{N}(i)} \right) \right)

    Args:
        aggr (Optional[Union[str, Aggregator]]): The aggregation method to use.
            (default: :obj:`"sum"`)
        aggr_kwargs (Optional[Dict[str, Any]]): Additional arguments for the aggregator.
            (default: :obj:`None`)
    """

    def __init__(
        self,
        aggr: Optional[Union[str, Aggregator]] = 'sum',
        *,
        aggr_kwargs: Optional[Dict[str, Any]] = None
    ):
        super().__init__()

        self.__explain__ = self.__is_overrided__(self.explain)
        self.__msg_aggr__ = self.__is_overrided__(self.message_and_aggregate)

        self.aggr_module = self.aggr_revoler(aggr, **(aggr_kwargs or {}))

    def propagate(
            self,
            x: Union[Tensor, Tuple[Tensor, Tensor]],
            edge_index: Union[Tensor, SparseTensor],
            **kwargs
    ) -> Tensor:
        r"""
        The initial call to start propagating messages.
        This method will call :meth:`message`, :meth:`aggregate` ( or :meth:`message_and_aggregate` if it's available )
        and :meth:`update` in sequence to complete once propagate.

        Args:
            x (Union[Tensor, Tuple[Tensor, Tensor]]):
                - `Tensor`: The input node feature matrix. :math:`(|V|, F_{in})`
                - `Tuple[Tensor, Tensor]`: The input node feature matrix for source and destination nodes.
            edge_index (Union[Tensor, SparseTensor]): The edge indices. Tensor, :math:`(2, |E|)`
            **kwargs: Additional arguments for the message, aggregate and update functions.
        """

        # Infer aggregator dim_size
        if 'dim_size' not in kwargs or kwargs['dim_size'] is None:
            if x is not None:
                if isinstance(x, Tensor):
                    kwargs['dim_size'] = x.size(0)
                else:
                    kwargs['dim_size'] = x[1].size(0)
            else:
                raise ValueError("dim_size must be provided while x is None.")

        # message and aggregate
        if self.__msg_aggr__:
            msg_aggr_kwargs = self.__collect__(
                self.message_and_aggregate, x, edge_index, kwargs
            )
            out = self.message_and_aggregate(**msg_aggr_kwargs)
        else:
            msg_kwargs = self.__collect__(
                self.message, x, edge_index, kwargs
            )
            out = self.message(**msg_kwargs)
            aggr_kwargs = self.__collect__(
                self.aggregate, x, edge_index, kwargs
            )
            out = self.aggregate(out, **aggr_kwargs)

        # update
        update_kwargs = self.__collect__(
            self.update, x, edge_index, kwargs
        )
        out = self.update(out, **update_kwargs)
        return out

    def message(
            self,
            x: Tensor,
            edge_index: Union[Tensor, SparseTensor],
            edge_weight: Tensor = None
    ) -> Tensor:
        r"""
        Compute message from src nodes :math:`v_j` to dst nodes :math:`v_i`.

        Args:
            x (Tensor): The input node feature matrix.
            edge_index (Union[Tensor, SparseTensor]): The edge indices or adj.
            edge_weight (Tensor): The edge weights.

        Returns:
            Tensor: The message tensor with size :math:`(|V_{src}|, \text{message_dim})`.
        """
        edge_index, edge_weight_ = self.__unify_edgeindex__(edge_index)
        edge_weight = edge_weight if edge_weight_ is None else edge_weight_
        src_index = edge_index[0, :]
        msgs = x.index_select(dim=0, index=src_index)

        if edge_weight is not None:
            return msgs * edge_weight.view(-1, 1)
        return msgs

    def aggregate(
        self,
        msgs: Tensor,
        edge_index: Union[Tensor, SparseTensor],
        dim: int = 0,
        dim_size: Optional[int] = None
    ):
        r"""
        Aggrate messages from src nodes to dst nodes.

        Args:
            msgs (Tensor): The messages to aggregate.
            edge_index (Union[Tensor, SparseTensor]): The edge indices.
            dim (int): The dimension to aggregate.
                (default: :obj:`0`)
            dim_size (Optional[int]): The size of output, tensor at dim. If None, infer from edge_index.
                (default: :obj:`None`)
        """
        edge_index, _ = self.__unify_edgeindex__(edge_index)
        return self.aggr_module(msgs, edge_index[1, :], dim=dim, dim_size=dim_size)

    def message_and_aggregate(self, edge_index: Union[Tensor, SparseTensor]) -> Tensor:
        r"""The message and aggregation interface to be overridden by subclasses."""
        return NotImplemented

    def update(self, output: Tensor) -> Tensor:
        r"""Update the dst node embeddings."""
        return output

    # Properties
    @property
    def if_message_and_aggregate(self) -> bool:
        return self.__msg_aggr__

    @if_message_and_aggregate.setter
    def if_message_and_aggregate(self, msg_aggr: bool) -> None:
        self.__msg_aggr__ = msg_aggr

    # Utility functions
    def __collect__(self, func: Callable, x, edge_index, kwargs) -> Dict[str, Any]:
        r"""Collects the arguments funcs.
        """
        func_params = OrderedDict(self.__func_params__(func))
        if func.__name__ in ['aggregate', 'update']:
            func_params.popitem(last=False)

        coll = OrderedDict()
        for k, v in func_params.items():
            if k in kwargs:
                coll[k] = kwargs[k]
            elif k == 'x':
                coll[k] = x
            elif k == 'edge_index':
                coll[k] = edge_index
            else:
                if v.default != inspect.Parameter.empty:
                    coll[k] = v.default
                else:
                    raise ValueError(f"Missing required parameter {k}.")
        return coll

    @lru_cache
    def __func_params__(self, func: Callable) -> OrderedDict:
        return inspect.signature(func).parameters

    @lru_cache
    def __unify_edgeindex__(self, edge_index: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Unify the edge index to a 2D tensor."""
        if edge_index.is_sparse:
            return self.__adj2edges__(edge_index)
        elif edge_index.size(0) != 2:
            try:
                return self.__adj2edges__(edge_index)
            except ValueError:
                raise ValueError(f"Expect edge_index to be a 2D tensor, got {edge_index.size()}.")
        else:
            return edge_index, None

    @lru_cache
    def __adj2edges__(self, adj: SparseTensor) -> Tuple[Tensor, Tensor]:
        r"""Converts a sparse adjacency matrix to edge indices."""
        if adj.is_sparse:
            coo_adj = adj.to_sparse_coo().coalesce()
            s, d, vs = coo_adj.indices()[0], coo_adj.indices()[1], coo_adj.values()
            vs = None if torch.all(vs == 1) else vs
            return torch.stack([s, d]), vs
        else:
            raise TypeError(f"Expect adj to be a SparseTensor, got {type(adj)}.")

    @lru_cache
    def __is_overrided__(self, func: Callable) -> bool:
        r"""Check if the function is overridden. If so, return True."""
        return getattr(self.__class__, func.__name__, None) \
            != getattr(MessagePassing, func.__name__)

    def aggr_revoler(self, target_aggr: Union[str, Aggregator], **kwargs) -> Aggregator:
        r"""Resolve the aggregator."""
        if isinstance(target_aggr, Aggregator):
            return target_aggr

        import rllm.nn.conv.graph_conv.aggrs as aggrs
        aggrs_l = [
            getattr(aggrs, name) for name in dir(aggrs)
            if inspect.isclass(getattr(aggrs, name))
        ]

        def normalize_str(s: str) -> str:
            return s.lower().replace('_', '').replace('-', '').replace(' ', '')

        norm_target_aggr = normalize_str(target_aggr)

        for aggr in aggrs_l:
            aggr_name = normalize_str(aggr.__name__)
            if norm_target_aggr in [aggr_name, aggr_name.replace('aggregator', '')]:
                return aggr(**kwargs)

        raise ValueError(f"Aggregator {target_aggr} not found.")

    @overload
    def retrieve_feats(
        self,
        feats: Tensor,
        edge_index: Union[Tensor, SparseTensor],
        dim: Optional[int] = None,
        retrieve_dim: int = 0
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        r"""Non-bipartite graph, :obj:`feats` contains all nodes' features.

        Args:
            feats (Tensor): The node features.
            edge_index (Union[Tensor, SparseTensor]): The edge indices.
            dim (Optionalp[int]): The edge_index dimension to retrieve. If None, retrieve both src and dst.
            retrieve_dim (int): The dimension to retrieve.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: Node features at dim, or both source and destination node features.
                :math:`(|E_{dim}|, F)` or (:math:`(|E_{dim}|, F)` and :math:`(|E_{dim}|, F)`).
        """
        ...

    @overload
    def retrieve_feats(
        self,
        feats: Tuple[Tensor, Tensor],
        edge_index: Union[Tensor, SparseTensor],
        dim: Optional[int] = None,
        retrieve_dim: int = 0
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        r"""Bipartite graph, :obj:`feats` contains source and destination nodes' features.

        Args:
            feats (Tensor): The node features.
            edge_index (Union[Tensor, SparseTensor]): The edge indices.
            dim (Optionalp[int]): The edge_index dimension to retrieve. If None, retrieve both src and dst.
            retrieve_dim (int): The dimension to retrieve.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: Node features at dim, or both source and destination node features.
                :math:`(|E_{dim}|, F)` or (:math:`(|E_{dim}|, F_{src})` and :math:`(|E_{dim}|, F_{dst})`).
        """
        ...

    def retrieve_feats(
        self,
        feats: Union[Tensor, Tuple[Tensor, Tensor]],
        edge_index: Union[Tensor, SparseTensor],
        dim: Optional[int] = None,
        retrieve_dim: int = 0
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:

        edge_index, _ = self.__unify_edgeindex__(edge_index)

        if isinstance(feats, tuple):
            src_feats, dst_feats = feats
            if dim is None:
                src_feats = src_feats.index_select(retrieve_dim, edge_index[0, :])
                dst_feats = dst_feats.index_select(retrieve_dim, edge_index[1, :])
                return src_feats, dst_feats
            else:
                assert dim in [0, 1], f"Expect dim to be 0 or 1, got {dim}."
                if dim == 0:
                    return src_feats.index_select(retrieve_dim, edge_index[0, :])
                else:
                    return dst_feats.index_select(retrieve_dim, edge_index[1, :])

        else:
            if dim is None:
                src_feats = feats.index_select(retrieve_dim, edge_index[0, :])
                dst_feats = feats.index_select(retrieve_dim, edge_index[1, :])
                return src_feats, dst_feats
            else:
                assert dim in [0, 1], f"Expect dim to be 0 or 1, got {dim}."
                return feats.index_select(retrieve_dim, edge_index[dim, :])

    # explain functions
    def explain(self, kwargs: Dict[str, Any]) -> Any:
        r"""Explain the behavior of the message passing layer.

        For now, keep it a interface and implement it in the subclasses if necessary.
        """
        raise NotImplementedError

    @property
    def if_explain(self) -> bool:
        r"""Whether to enable explain mode."""
        return self.__explain__

    @if_explain.setter
    def if_explain(self, explain: bool) -> None:
        r"""Set the explain mode."""
        self.__explain__ = explain
