from __future__ import annotations

import torch.nn.functional as F

from rllm.types import ColType, StatType
from rllm.data import TableData
from .col_transform import ColTransform


class OneHotTransform(ColTransform):
    r"""A simple one-hot encoding Transform for categorical features.
    The OneHotTransform class is designed to perform one-hot encoding on
    categorical features in tabular data. This transformation converts
    categorical feature values into a multidimensional tensor representation.

    Args:
        out_dim (int, optional):
            The output dimensionality for the one-hot encoded features. If set
            to 0, the dimensionality will be determined by the number of
            unique categories in the data (default: 0).
    """

    def __init__(
        self,
        out_dim: int = 0,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim

    def forward(
        self,
        data: TableData,
    ) -> TableData:
        if ColType.CATEGORICAL in data.feat_dict.keys():
            stat_list = data.metadata[ColType.CATEGORICAL]
            feat = data.feat_dict[ColType.CATEGORICAL]

            # Determine the number of categories
            # If out_dim is not specified, use the maximum number of categories
            # If out_dim is specified, use the maximum of the specified value
            # and the number of categories
            self.num_categories = max([stats[StatType.COUNT] for stats in stat_list])
            one_hot_classes = (
                self.num_categories
                if self.num_categories > self.out_dim
                else self.out_dim
            )

            one_hot_feat = F.one_hot(feat, num_classes=one_hot_classes)
            data.feat_dict[ColType.CATEGORICAL] = one_hot_feat

        return data
