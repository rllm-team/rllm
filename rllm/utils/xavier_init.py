import math

import torch
from torch import Tensor


def _xavier_uniform_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    r"""Xavier uniform initialization that operates on the last two dims.

    Unlike :func:`torch.nn.init.xavier_uniform_`, this supports parameters with
    a leading group dimension (e.g. :obj:`[num_metapaths, cin, cout]`) by using
    the trailing two dimensions to compute the fan-in/fan-out.

    Args:
        tensor (Tensor): Parameter to be initialized.
        gain (float): Gain factor. (default: :obj:`1.0`)
        
    Returns:
        Tensor: The initialized parameter.
    """
    fan_in, fan_out = tensor.size()[-2:]
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std
    return torch.nn.init._no_grad_uniform_(tensor, -a, a)
