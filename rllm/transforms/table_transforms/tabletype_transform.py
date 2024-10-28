from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.nn import Module, ModuleDict

from rllm.types import ColType, StatType
from rllm.transforms.table_transforms import ColTypeTransform


class TableTypeTransform(Module):
    r"""Table Transform that transforms each ColType tensor into embeddings and
    performs the final concatenation.

    Args:
        out_dim (int): Output dimensionality.
        col_stats_dict
            (Dict[class:`rllm.types.ColType`, List[dict[StatType, Any]]):
            A dictionary that maps column type into stats.
        col_types_transform_dict
            (Dict[:class:`rllm.types.ColType`,
            :class:`rllm.nn.encoder.ColTypeTransform`]):
            A dictionary that maps :class:`rllm.types.ColType` into
            :class:`rllm.nn.encoder.ColTypeTransform` class. Only
            parent :class:`stypes <rllm.types.ColType>` are supported
            as keys.
    """
    def __init__(
        self,
        out_dim: int,
        col_stats_dict: dict[ColType, list[dict[str, Any]]],
        col_types_transform_dict: dict[ColType, ColTypeTransform],
    ) -> None:
        super().__init__()

        self.col_stats_dict = col_stats_dict
        self.transform_dict = ModuleDict()

        col_names_dict: dict[ColType, list[str]] = {}
        for col_type, stats_list in col_stats_dict.items():
            if col_type not in col_names_dict.keys():
                col_names_dict[col_type] = []
            for stats in stats_list:
                col_names_dict[col_type].append(stats[StatType.COLNAME])
        self.col_names_dict = col_names_dict

        for col_type, col_types_transform in col_types_transform_dict.items():
            if col_type not in col_types_transform.supported_types:
                raise ValueError(
                    f"{col_types_transform} does not "
                    f"support encoding {col_type}."
                )

            if col_type in col_stats_dict.keys():
                stats_list = col_stats_dict[col_type]
                # Set attributes
                col_types_transform.col_type = col_type
                if col_types_transform.out_dim is None:
                    col_types_transform.out_dim = out_dim
                col_types_transform.stats_list = stats_list
                self.transform_dict[col_type.value] = col_types_transform
                col_types_transform.post_init()

    def reset_parameters(self):
        for col_type in self.col_stats_dict.keys():
            self.transform_dict[col_type.value].reset_parameters()

    def forward(
        self,
        feat_dict: dict[ColType, Tensor]
    ) -> tuple[Tensor, list[str]]:
        all_col_names = []
        xs = []
        for col_type in feat_dict.keys():
            feat = feat_dict[col_type]
            col_names = self.col_names_dict[col_type]
            x = self.transform_dict[col_type.value](feat, col_names)
            xs.append(x)
            all_col_names.extend(col_names)
        x = torch.cat(xs, dim=1)
        return x, all_col_names
