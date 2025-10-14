import torch
from torch import Tensor


def sanitize_name(arcname: str, pathsep: str):
    """Replace bad characters and remove trailing dots from parts."""
    illegal = ':<>|"?*'
    table = str.maketrans(illegal, '_' * len(illegal))
    arcname = arcname.translate(table)
    # remove trailing dots
    arcname = (x.rstrip('.') for x in arcname.split(pathsep))
    # rejoin, removing empty parts.
    arcname = pathsep.join(x for x in arcname if x)
    return arcname


def index2mask(index: Tensor, size: int):
    r"""Convert index to mask format."""
    mask = torch.zeros(size, dtype=bool)
    mask[index] = True
    return mask
