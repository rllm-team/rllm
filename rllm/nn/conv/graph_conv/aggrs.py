from typing import Optional

import torch
from torch import Tensor


class Aggregator(torch.nn.Module):
    r"""Base class for Aggregator.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        dim: int = 0,
        dim_size: Optional[int] = None,
    ) -> Tensor:
        """
        Args:
            x (Tensor): The input tensor.
            index (Tensor): The index tensor.
            dim (int): The dimension to index.
            dim_size (int, optional): The size of the output tensor at dimension dim.
        """
        raise NotImplementedError

    def reset_parameters(self) -> None:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    # util funcs
    def reduce(
        self,
        x: Tensor,
        index: Tensor,
        dim: int = 0,
        dim_size: Optional[int] = None,
        reduce: str = 'sum'
    ) -> Tensor:
        r"""
        Reduce the tensor x to the shape of (dim_size, x.size(1)) by index.

        Args:
            x (Tensor): The input tensor.
            index (Tensor): The index tensor.
            dim (int): The dimension to index and reduce.
            dim_size (int, optional): The size of the output tensor at dimension dim.
                If set to None, will create a minimal-sized output tensor according to index.max() + 1.
            reduce (str): The reduce method. {'sum (add)', 'mean', 'max', 'min', 'prod (mul)'}
        """
        if dim_size is None:
            dim_size = index.max().item() + 1

        out_size = list(x.size())
        out_size[dim] = dim_size
        output: Tensor = torch.zeros(out_size, dtype=x.dtype, device=x.device)

        reduce = reduce.lower()
        reduce = 'prod' if reduce == 'mul' else reduce

        if reduce == 'add' or reduce == 'sum':
            return output.index_add_(dim, index, x)
        elif reduce == 'mean' or reduce == 'prod':
            return torch.scatter_reduce(
                output,
                dim=dim,
                index=index.unsqueeze(-1).expand_as(x),
                src=x,
                reduce=reduce,
                include_self=False
            )
        elif reduce == 'max' or reduce == 'min':
            reduce = 'amin' if reduce == 'min' else 'amax'
            return torch.scatter_reduce(
                output,
                dim=dim,
                index=index.unsqueeze(-1).expand_as(x),
                src=x,
                reduce=reduce,
                include_self=False
            )
        else:
            raise ValueError(f"Reduce method {reduce} not supported.")

    def to_dense_batch(
        self,
        x: Tensor,
        index: Tensor,  # batch index
        batch_size: Optional[int] = None,
        max_num_nodes: Optional[int] = None,
        fill_value: float = 0.0
    ):
        r"""Transform input tensor to a dense batch tensor via index.

        Args:
            x (Tensor): The input tensor.
            index (Tensor): The batch index tensor.
            batch_size (int, optional): The number of batches. If None, it will be inferred from index.
            max_num_nodes (int, optional): The maximum number of nodes in a batch.
            fill_value (float, optional): The value for invalid entries in the batch matrix.
        """
        if index is None and max_num_nodes is None:
            mask = torch.ones((1, x.size(0)), dtype=torch.bool, device=x.device)
            return x.unsqueeze(0), mask

        if index is None:
            index = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        if batch_size is None:
            batch_size = int(index.max().item() + 1)

        batch_num_nodes = index.new_zeros(index.max().item() + 1)
        batch_num_nodes = torch.scatter_add(
            batch_num_nodes,
            dim=0,
            index=index,
            src=index.new_ones(x.size(0))
        )

        cum_nodes = torch.cat([index.new_zeros(1), batch_num_nodes.cumsum(dim=0)])

        filter_nodes = False

        if max_num_nodes is None:
            max_num_nodes = int(batch_num_nodes.max().item())
        else:
            filter_nodes = max_num_nodes < int(batch_num_nodes.max().item())

        # tmp: node idx in its batch
        # idx: node idx in out tensor
        tmp = torch.arange(index.size(0), device=x.device) - cum_nodes[index]
        idx = tmp + (index * max_num_nodes)

        if filter_nodes:
            mask = tmp < max_num_nodes
            x, idx = x[mask], idx[mask]

        size = [batch_size * max_num_nodes] + list(x.size())[1:]
        out = torch.as_tensor(fill_value, dtype=x.dtype, device=x.device)
        out = out.repeat(size)
        out[idx] = x
        out = out.view([batch_size, max_num_nodes] + list(x.size())[1:])

        mask = torch.zeros(batch_size * max_num_nodes, dtype=torch.bool, device=x.device)
        mask[idx] = 1
        mask = mask.view(batch_size, max_num_nodes)

        return out, mask


class MeanAggregator(Aggregator):
    r"""Mean Aggregator for Graph Convolutional Networks.
    """
    def forward(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        dim: int = 0,
        dim_size: Optional[int] = None,
    ) -> Tensor:
        return self.reduce(x, index, dim=dim, dim_size=dim_size, reduce='mean')


class MaxAggregator(Aggregator):
    r"""Max Aggregator for Graph Convolutional Networks.
    """
    def forward(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        dim: int = 0,
        dim_size: Optional[int] = None,
    ) -> Tensor:
        return self.reduce(x, index, dim=dim, dim_size=dim_size, reduce='max')


class MinAggregator(Aggregator):
    r"""Min Aggregator for Graph Convolutional Networks.
    """
    def forward(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        dim: int = 0,
        dim_size: Optional[int] = None,
    ) -> Tensor:
        return self.reduce(x, index, dim=dim, dim_size=dim_size, reduce='min')


class SumAggregator(Aggregator):
    r"""Sum(GCN, Add) Aggregator for Graph Convolutional Networks.
    """

    def forward(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        dim: int = 0,
        dim_size: Optional[int] = None,
    ) -> Tensor:
        return self.reduce(x, index, dim=dim, dim_size=dim_size, reduce='sum')


class AddAggregator(SumAggregator):
    pass


class GCNAggregator(SumAggregator):
    pass


class ProdAggregator(Aggregator):
    r"""Prod Aggregator for Graph Convolutional Networks.
    """

    def forward(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        dim: int = 0,
        dim_size: Optional[int] = None,
    ) -> Tensor:
        return self.reduce(x, index, dim=dim, dim_size=dim_size, reduce='prod')


class MaxPoolAggregator(Aggregator):
    r"""Max Pool Aggregator for Graph Convolutional Networks.

    x -> MLP -> Max Pooling

    Args:
        in_dim (int): The input feature dimension.
        out_dim (int): The output feature dimension.
        dropout (float, optional): The
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.mlp:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        dim: int = 0,
        dim_size: Optional[int] = None,
    ) -> Tensor:
        x = self.mlp(x)
        return self.reduce(x, index, dim=dim, dim_size=dim_size, reduce='max')

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"in_dim={self.in_dim}, "
                f"out_dim={self.out_dim}, "
                f"dropout={self.dropout})")


class MeanPoolAggregator(Aggregator):
    r"""Mean Pool Aggregator for Graph Convolutional Networks.

    x -> MLP -> Mean Pooling

    Args:
        in_dim (int): The input feature dimension.
        out_dim (int): The output feature dimension.
        dropout (float, optional): The dropout rate.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.mlp:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        dim: int = 0,
        dim_size: Optional[int] = None,
    ) -> Tensor:
        x = self.mlp(x)
        return self.reduce(x, index, dim=dim, dim_size=dim_size, reduce='mean')

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"in_dim={self.in_dim}, "
                f"out_dim={self.out_dim}, "
                f"dropout={self.dropout})")


class LSTMAggregator(Aggregator):
    r"""LSTM Aggregator for Graph Convolutional Networks.

    Args:
        in_dim (int): The input feature dimension.
        out_dim (int): The output feature dimension.
        **kwargs: Other arguments for torch.nn.LSTM.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        **kwargs
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lstm = torch.nn.LSTM(in_dim, out_dim, batch_first=True, **kwargs)  # (batch, seq, feature)
        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.reset_parameters()

    def forward(
        self,
        x: Tensor,
        index: Tensor,
        dim: int = 0,
        dim_size: Optional[int] = None,
        max_num_nodes: Optional[int] = None,
    ) -> Tensor:
        assert torch.all(index[:-1] <= index[1:]), "Index must be sorted."
        assert dim == 0, "Only support node dim=0."

        x, _ = self.to_dense_batch(x, index, batch_size=dim_size, max_num_nodes=max_num_nodes)
        return self.lstm(x)[0][: -1].squeeze()  # remove padding

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"in_dim={self.in_dim}, "
                f"out_dim={self.out_dim})")
