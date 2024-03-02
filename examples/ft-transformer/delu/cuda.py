"""An extension to `torch.cuda`."""
import gc

import torch

__all__ = ['free_memory']


def free_memory() -> None:
    """Free GPU memory = `torch.cuda.synchronize` + `gc.collect` + `torch.cuda.empty_cache`.

    .. note::
        There is a small chunk of GPU-memory (occupied by drivers) that is impossible to
        free. This is a property of `torch`, so this function inherits this property.

    **Usage**

    >>> delu.cuda.free_memory()
    """  # noqa: E501
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # type: ignore
    gc.collect()
    torch.cuda.empty_cache()
