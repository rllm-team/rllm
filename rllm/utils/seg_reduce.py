import torch
from torch import Tensor


def seg_sum(data: Tensor, segment_ids: Tensor, num_segments: int):
    r"""Computes the sum of elements in `data` for each segment
    specified by `segment_ids`.

    Args:
        data (Tensor): A tensor, typically two-dimensional.
        segment_ids (Tensor): A one-dimensional tensor that indicates the
            segmentation in data.
        num_segments (int): Total segments.

    Returns:
        output: sum calculated by segment_ids with the same shape as data.
    """
    output = torch.zeros(
        (num_segments, data.size(1)), device=data.device, dtype=data.dtype
    )
    return torch.scatter_reduce(
        output,
        dim=0,
        index=segment_ids.unsqueeze(1).expand(-1, data.size(1)),
        src=data,
        reduce="sum",
    )


def seg_softmax(data: Tensor, segment_ids: Tensor, num_segs: int):
    r"""Computes the softmax scores of elements in `data` for each segment
    specified by `segment_ids`.

    Args:
        data (Tensor): A tensor, typically two-dimensional.
        segment_ids (Tensor): A one-dimensional tensor that indicates the
            segmentation in data.
        num_segments (int): Total segments.

    Returns:
        score: softmax score, which has the same shape as data.
    """
    max_values = torch.zeros(
        num_segs, data.size(1), device=data.device, dtype=data.dtype
    )
    max_values = torch.scatter_reduce(
        max_values,
        dim=0,
        index=segment_ids.unsqueeze(1).expand(-1, data.size(1)),
        src=data,
        reduce="amax",
    )
    gathered_max_values = max_values[segment_ids]
    exp = torch.exp(data - gathered_max_values)

    denominator = torch.zeros(num_segs, data.size(1), device=data.device)
    denominator = torch.scatter_reduce(
        denominator,
        dim=0,
        index=segment_ids.unsqueeze(1).expand(-1, data.size(1)),
        src=exp,
        reduce="sum",
    )
    gathered_denominator = denominator[segment_ids]
    score = exp / (gathered_denominator + 1e-16)
    return score


def seg_softmax_(data: Tensor, segment_ids: Tensor, num_segs: int):
    r"""Computes the softmax scores of elements in `data` for each segment
    specified by `segment_ids`.

    Args:
        data (Tensor): A tensor, typically two-dimensional.
        segment_ids (Tensor): A one-dimensional tensor that indicates the
            segmentation in data.
        num_segments (int): Total segments.

    Returns:
        score: softmax score, which has the same shape as data.
    """
    max_values = torch.zeros(
        num_segs, data.size(1), device=data.device, dtype=data.dtype
    )
    for i in range(num_segs):
        segment_data = data[segment_ids == i]
        if segment_data.size(0) > 0:
            max_values[i] = segment_data.max(dim=0)[0]

    gathered_max_values = max_values[segment_ids]  # (E, H)
    exp = torch.exp(data - gathered_max_values)  # (E, H)

    denominator = torch.zeros(num_segs, data.size(1), device=data.device)
    for i in range(num_segs):
        segment_exp = exp[segment_ids == i]
        if segment_exp.size(0) > 0:
            denominator[i] = segment_exp.sum(dim=0)

    gathered_denominator = denominator[segment_ids]  # (E, H)
    score = exp / (gathered_denominator + 1e-16)  # (E, H)
    return score
