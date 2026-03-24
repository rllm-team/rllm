import torch
from torch import Tensor


def sanitize_name(arcname: str, pathsep: str):
    r"""Sanitize archive member names for safe filesystem usage.

    This function replaces illegal filename characters and removes trailing
    dots from each path component.

    Args:
        arcname (str): Original archive member path.
        pathsep (str): Path separator used by the archive path.

    Returns:
        str: Sanitized relative path.

    Examples:
        >>> sanitize_name("a:bad|name./b.", "/")
        'a_bad_name/b'
    """
    illegal = ':<>|"?*'
    table = str.maketrans(illegal, '_' * len(illegal))
    arcname = arcname.translate(table)
    # remove trailing dots
    arcname = (x.rstrip('.') for x in arcname.split(pathsep))
    # rejoin, removing empty parts.
    arcname = pathsep.join(x for x in arcname if x)
    return arcname


def index_to_mask(index: Tensor, size: int):
    r"""Convert index tensor to a boolean mask tensor.

    Args:
        index (Tensor): Index tensor containing selected positions.
        size (int): Total number of positions in the output mask.

    Returns:
        Tensor: Boolean mask of shape ``[size]``.

    Examples:
        >>> index = torch.tensor([0, 2, 4])
        >>> index_to_mask(index, 6)
        tensor([ True, False,  True, False,  True, False])
    """
    mask = torch.zeros(size, dtype=bool)
    mask[index] = True
    return mask
