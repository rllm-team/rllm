from abc import ABC, abstractmethod
import inspect
from collections import OrderedDict
from typing import List, Tuple, Callable, Dict, Any, Union, Optional

import torch
from torch import Tensor
from torch.sparse import Tensor as SparseTensor


class MessagePassing(torch.nn.Module, ABC):
    r"""Base class for message passing.
    """

    special_args = {
        "x",
        "edge_index",
        "edge_weight",
    }

    def __init__(self):
        super().__init__()
        self.__explain__ = False

    def propagate(
            self,
            x: Tensor,
            edge_index: Union[Tensor, SparseTensor],
            num_nodes: Optional[int] = None,
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
        if num_nodes is None:
            num_nodes = x.size(0)

        kwargs = self.__collect__(x, edge_index, num_nodes, kwargs)
        

        pass

    def message(self, x_j: Tensor) -> Tensor:
        r"""
        Compute message from src nodes :math:`v_j` to dst nodes :math:`v_i`.

        Params like :obj:`x_j` and :obj:`x_i` are symbolic representation of src
        and dst nodes feature tensors of real node feature tensors like :obj:`x`
        in :obj:`propagate` function.

        Returns:
            Tensor: The message tensor with size :math:`(|V_{src}|, \text{message_dim})`.
        """
        return x_j

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
        - sum
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
        dst_index = edge_index[1, :]
        if num_nodes is None:
            num_nodes = dst_index.max().item() + 1
        else:
            assert num_nodes >= dst_index.max().item() + 1, (
                f"Number of nodes should be equal or greater than {dst_index.max().item() + 1}"
            )
        output: Tensor = torch.zeros(num_nodes, dtype=msgs.dtype)
        if aggr == 'add':
            return output.scatter_add_(0, dst_index, msgs)
        elif aggr == 'mean':
            cnt = torch.zeros(num_nodes, dtype=msgs.dtype).scatter_add_(0, dst_index, torch.ones_like(msgs))
            return output.scatter_add_(0, dst_index, msgs) / cnt
        elif aggr == 'max':
            return output.scatter_(0, dst_index, msgs, reduce='amax')
        else:
            raise ValueError(f"Aggregation method {aggr} not supported.")

    def message_and_aggregate(self, edge_index: Union[Tensor, SparseTensor]):
        r"""The message and aggregation interface to be overridden by subclasses."""
        return NotImplemented

    def update(self, output: Tensor) -> Tensor:
        r"""Update the dst node embeddings."""
        return output

    # Utility functions
    def __collect__(self, x, edge_index, num_nodes, kwargs) -> Dict[str, Any]:
        r"""Collects the arguments for message function."""

        pass

    def __func_params__(self, func: Callable) -> OrderedDict:
        return inspect.signature(func).parameters

    def __dispatch_params__(self, func: Callable, kwargs: Dict[str, Any]) -> None:
        r"""Dispatches the arguments of a given function based on its
        signature and the given keyword arguments.
        """
        pass
    
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
