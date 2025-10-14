from typing import List

from torch import Tensor


def lexsort(
    keys: List[Tensor],
    dim: int = -1,
    descending: bool = False,
) -> Tensor:
    r"""Perform an indirect stable sort using a sequence of keys.

    Given multiple sorting keys, lexsort returns an array of integer indices
    that describes the sort order by multiple keys. The last key in the
    sequence is used for the primary sort order,
    ties are broken by the second-to-last key, and so on.

    Example:
        >>> a = torch.tensor([1, 5, 1, 4, 3, 4, 4])  # First sequence
        >>> b = torch.tensor([9, 4, 0, 4, 0, 2, 1])  # Second sequence
        >>> ind = lexsort((b, a))  # Sort by `a`, then by `b`
        >>> ind
        tensor([2, 0, 4, 6, 5, 3, 1])
        >>> [torch.tensor((a[i], b[i])) for i in ind]
        [tensor([1, 0]), tensor([1, 9]), tensor([3, 0]), tensor([4, 1]),
        tensor([4, 2]), tensor([4, 4]), tensor([5, 4])]

    Args:
        keys (sequence of Tensors): keys to sort by, sort from last to first.
        dim (int, optional): the dimension along which to sort. Default is -1.
        descending (bool, optional): controls the sorting order (ascending or descending).
            Default is False.
    """
    out = keys[0].argsort(dim=dim, descending=descending, stable=True)
    for k in keys[1:]:
        index = k.gather(dim=dim, index=out)
        index = index.argsort(dim=dim, descending=descending, stable=True)
        out = out.gather(dim=dim, index=index)
    return out
