from __future__ import annotations
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor
from torch.nn import Parameter, ParameterList

from rllm.types import ColType, StatType
from ._col_encoder import ColEncoder


class TextEmbeddingEncoder(ColEncoder):
    r"""A text embedding encoder for embedded text columns.
    It applies a linear layer to each text column
    and concatenates the output embeddings.

    Args:
        out_dim (int, optional): The output dimensionality (default: :obj:`1`).
        stats_list (List[Dict[rllm.types.StatType, Any]], optional): The list of statistics
            for each column within the same column type (default: :obj:`None`).
        post_module (torch.nn.Module, optional): The post-hoc module applied to the
            output, such as activation function and normalization. Must
            preserve the shape of the output. If :obj:`None`, no module will
            be applied to the output (default: :obj:`None`).
    """

    supported_types = {ColType.TEXT}

    def __init__(
        self,
        out_dim: Optional[int] = None,
        stats_list: Optional[List[Dict[StatType, Any]]] = None,
        post_module: Optional[torch.nn.Module] = None,
    ):
        super().__init__(out_dim, stats_list, post_module)

    def post_init(self):
        num_cols = len(self.stats_list)
        self.emb_dim_list = [stats[StatType.EMB_DIM] for stats in self.stats_list]
        # W list: [D, out_dim] * num_cols
        self.weight_list = ParameterList(
            [
                Parameter(torch.empty(emb_dim, self.out_dim))
                for emb_dim in self.emb_dim_list
            ]
        )
        # B: [num_cols, out_dim]
        self.biases = Parameter(torch.empty(num_cols, self.out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        for weight in self.weight_list:
            torch.nn.init.normal_(weight, std=0.01)
        torch.nn.init.zeros_(self.biases)

    def encode_forward(self, feat: Tensor) -> Tensor:
        # feat: [B, num_cols, D]
        x_lins: list[Tensor] = []
        col_idx = 0
        for idx in range(len(self.emb_dim_list)):
            # [B, D] * [D, out_dim]
            # -> [B, out_dim]
            x_lin = feat[:, col_idx, :] @ self.weight_list[idx]
            x_lins.append(x_lin)
            col_idx += 1
        # [B, num_cols, out_dim]
        x = torch.stack(x_lins, dim=1)
        # [B, num_cols, out_dim] + [num_cols, out_dim]
        # -> [B, num_cols, out_dim]
        x = x + self.biases
        return x
