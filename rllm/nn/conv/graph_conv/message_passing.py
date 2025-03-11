from abc import ABC
import inspect
from collections import OrderedDict
from functools import lru_cache
from typing import Tuple, Callable, Dict, Any, Union, Optional

import torch
from torch import Tensor
from torch.sparse import Tensor as SparseTensor

from rllm.nn.conv.graph_conv.aggrs import Aggregator


class MessagePassing(torch.nn.Module, ABC):
    r"""Base class for message passing.
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
            x: Tensor,
            edge_index: Union[Tensor, SparseTensor],
            **kwargs
    ) -> Tensor:
        r"""
        The initial call to start propagating messages.
        This method will call :meth:`message`, :meth:`aggregate` ( or :meth:`message_and_aggregate` if it's available )
        and :meth:`update` in sequence to complete once propagate.

        Args:
            x (Tensor): The input node feature matrix. :math:`(|V|, F_{in})`
            edge_index (Union[Tensor, SparseTensor]): The edge indices. Tensor, :math:`(2, |E|)`
        """
        # if 'num_nodes' not in kwargs or kwargs['num_nodes'] is None:
        #     kwargs['num_nodes'] = x.size(0)

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
            dim_size (Optional[int]): The size of output tensor at dim.
                (default: :obj:`None`)
        """
        edge_index, _ = self.__unify_edgeindex__(edge_index)
        return self.aggr_module(msgs, edge_index[1:].squeeze(), dim=dim, dim_size=dim_size)

    def aggregate_(
            self,
            msgs: Tensor,
            edge_index: Union[Tensor, SparseTensor],
            num_nodes: Optional[int],
            aggr: str = 'sum'
    ) -> Tensor:
        r"""
        Deprecated. Use :meth:`aggregate` instead.
        Aggrate messages from src nodes :math:`v_j` to dst nodes :math:`v_i`, i.e.
        compute the new features for each node by aggregating its neighbors' messages.

        Default supported aggregation methods:
        - sum (add)
        - mean
        - max

        Args:
            msgs (Tensor): The messages to aggregate.
            edge_index (Union[Tensor, SparseTensor]): The edge indices.
            num_nodes (Optional[int]): The number of nodes.
            aggr (str): The aggregation method to use (default: 'sum').
                Default supported options: 'sum', 'mean', 'max'.
        """
        edge_index, _ = self.__unify_edgeindex__(edge_index)
        dst_index = edge_index[1, :]
        output: Tensor = torch.zeros((num_nodes, msgs.size(1)), dtype=msgs.dtype, device=msgs.device)
        if aggr == 'add' or aggr == 'sum':
            return output.index_add_(0, dst_index, msgs)
        elif aggr == 'mean':
            count = torch.zeros(num_nodes, dtype=msgs.dtype, device=msgs.device)
            count.index_add_(0, dst_index, torch.ones_like(dst_index, dtype=msgs.dtype))
            count = torch.where(count == 0, torch.tensor(1e-10, dtype=count.dtype, device=count.device), count)
            output.index_add_(0, dst_index, msgs)
            return output / count.unsqueeze(-1)
        elif aggr == 'max':
            return output.scatter_(0, dst_index.unsqueeze(-1).expand_as(msgs), msgs)
        else:
            raise ValueError(f"Aggregation method {aggr} not supported.")

    def message_and_aggregate(self, edge_index: Union[Tensor, SparseTensor]) -> Tensor:
        r"""The message and aggregation interface to be overridden by subclasses."""
        return NotImplemented

    def update(self, output: Tensor) -> Tensor:
        r"""Update the dst node embeddings."""
        return output

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

    @property
    def if_message_and_aggregate(self) -> bool:
        return self.__msg_aggr__

    @if_message_and_aggregate.setter
    def if_message_and_aggregate(self, msg_aggr: bool) -> None:
        self.__msg_aggr__ = msg_aggr

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
