from abc import ABC
import inspect
from collections import OrderedDict
from functools import lru_cache
from typing import Tuple, Callable, Dict, Any, Union, Optional

import torch
from torch import Tensor
from torch.sparse import Tensor as SparseTensor


class MessagePassing(torch.nn.Module, ABC):
    r"""Base class for message passing.
    """

    def __init__(self):
        super().__init__()

        self.__explain__ = self.__is_overrided__(self.explain)
        self.__msg_aggr__ = self.__is_overrided__(self.message_and_aggregate)

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
        num_nodes = kwargs['num_nodes'] if ('num_nodes' in kwargs and kwargs['num_nodes'] is not None) else x.size(0)

        if self.__msg_aggr__:
            msg_aggr_kwargs = self.__collect__(
                self.message_and_aggregate, x, edge_index, kwargs
            )
            out = self.message_and_aggregate(**msg_aggr_kwargs)
            out = self.update(out)
            return out
        else:
            msg_kwargs = self.__collect__(
                self.message, x, edge_index, kwargs
            )
            out = self.message(**msg_kwargs)
            out = self.aggregate(out, edge_index, num_nodes, kwargs.get('aggr', 'sum'))
            out = self.update(out)
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
        if edge_index.is_sparse:
            edge_index, edge_weight = self.__adj2edges__(edge_index)
        src_index = edge_index[0, :]
        msgs = x.index_select(dim=0, index=src_index)

        if edge_weight is not None:
            return msgs * edge_weight.view(-1, 1)
        return msgs

    def aggregate(
            self,
            msgs: Tensor,
            edge_index: Union[Tensor, SparseTensor],
            num_nodes: Optional[int] = None,
            aggr: str = 'sum'
    ) -> Tensor:
        r"""
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
        # TODO: `scatter_add_` behaves nondeterministically, need to find a substitute.
        if edge_index.is_sparse:
            edge_index, _ = self.__adj2edges__(edge_index)
        dst_index = edge_index[1, :]
        if num_nodes is None:
            num_nodes = dst_index.max().item() + 1
        else:
            assert num_nodes >= dst_index.max().item() + 1, (
                f"Number of nodes should be equal or greater than {dst_index.max().item() + 1}"
            )
        # output: Tensor = torch.zeros(num_nodes, dtype=msgs.dtype)
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
        r"""Collects the arguments for message and message_and_aggregate funcs.
        """
        coll = OrderedDict()
        for k, v in self.__func_params__(func).items():
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

    def __func_params__(self, func: Callable) -> OrderedDict:
        return inspect.signature(func).parameters

    @lru_cache
    def __adj2edges__(self, adj: SparseTensor) -> Tuple[Tensor, Tensor]:
        r"""Converts a sparse adjacency matrix to edge indices."""
        if adj.is_sparse:
            coo_adj = adj.to_sparse_coo().coalesce()
            s, d, vs = coo_adj.indices()[0], coo_adj.indices()[1], coo_adj.values()
            return torch.stack([s, d]), vs
        else:
            raise TypeError(f"Expect adj to be a SparseTensor, got {type(adj)}.")

    def __is_overrided__(self, func: Callable) -> bool:
        r"""Check if the function is overridden. If so, return True."""
        return getattr(self.__class__, func.__name__, None) \
            != getattr(MessagePassing, func.__name__)

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
