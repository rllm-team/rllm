from typing import Union

import numpy as np
import torch
from torch import Tensor
import torch.utils
import torch.utils.data

from rllm.data import TableData
from ._torch_feature import WITH_TORCH_20


def be_mem_share_index_select(
    value: Union[np.ndarray, Tensor, TableData],
    index: Tensor,
    dim: int = 0,
):
    r"""Best-effort memory-sharing index select for use with
    :class:`~torch.utils.data.DataLoader`.

    In multi-process data loading, each worker normally creates a copy of
    the data before passing it to the main process. This function attempts
    to have workers and the main process share the same underlying memory to
    avoid redundant copies.

    Args:
        value (Union[numpy.ndarray, Tensor, TableData]): The source data to
            index into.
        index (Tensor): A 1-D tensor of integer indices.
        dim (int): The dimension along which to index.
            (default: :obj:`0`)

    Returns:
        Union[Tensor, TableData]: The indexed result.
    """
    assert index.dim() == 1, f"Index should be 1D, but {index.dim()}D found."

    if isinstance(value, Tensor):
        if torch.utils.data.get_worker_info() is not None:
            # In worker process, we create a shared memory tensor
            size = list(value.shape)
            size[dim] = index.numel()
            numel = np.prod(size)
            if WITH_TORCH_20:
                store = value.untyped_storage()._new_shared(
                    numel * value.element_size()
                )
                out = value.new(store).view(size)
                return torch.index_select(value, dim, index, out=out)
            else:
                raise NotImplementedError(
                    "`mem_share_index_select` is not supported for torch < 2.0."
                    "Check your torch version."
                )
        else:
            return torch.index_select(value, dim, index)

    if isinstance(value, TableData):
        assert dim == 0, "TableData only support dim=0."
        return value[index]

    if isinstance(value, np.ndarray):
        return torch.from_numpy(np.take(value, index.numpy(), axis=dim))

    raise ValueError(f"Unsupported type {type(value)}.")
