from __future__ import annotations

from rllm.types import ColType
from rllm.data import TableData
from .col_transform import ColTransform


class StackNumerical(ColTransform):
    r"""The StackNumerical class is designed to transform numerical features
    in tabular data by stacking them into a specified dimension. This
    transformation changes the shape of the numerical features from
    [batch_size, num_feats] to [batch_size, num_feats, out_dim], effectively
    replicating the values along the new dimension.

    Args:
        out_dim (int): The output dimensionality to which the numerical
            features will be stacked.
    """

    def __init__(
        self,
        out_dim: int,
    ) -> None:
        self.out_dim = out_dim

    def forward(
        self,
        data: TableData,
    ) -> TableData:
        if ColType.NUMERICAL in data.feat_dict.keys():
            feat = data.feat_dict[ColType.NUMERICAL]
            data.feat_dict[ColType.NUMERICAL] = feat.unsqueeze(2).repeat(
                1, 1, self.out_dim
            )
        return data
