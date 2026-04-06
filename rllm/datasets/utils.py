import torch
from torch import Tensor


def sanitize_name(arcname: str, pathsep: str) -> str:
    r"""Sanitize archive path names for cross-platform filesystem safety.

    This utility replaces characters that are invalid on Windows and trims
    trailing dots from each path component.

    Args:
        arcname (str): Original archive entry path.
        pathsep (str): Path separator used inside the archive (for example,
            :obj:`"/"`).
    """
    illegal = ':<>|"?*'
    table = str.maketrans(illegal, '_' * len(illegal))
    arcname = arcname.translate(table)
    # remove trailing dots
    arcname = (x.rstrip('.') for x in arcname.split(pathsep))
    # rejoin, removing empty parts.
    arcname = pathsep.join(x for x in arcname if x)
    return arcname


def index_to_mask(index: Tensor, size: int) -> Tensor:
    r"""Convert index tensor to a boolean mask.

    Args:
        index (Tensor): 1D index tensor containing selected positions.
        size (int): Length of the output mask.
    """
    mask = torch.zeros(size, dtype=bool)
    mask[index] = True
    return mask
