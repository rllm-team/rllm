from __future__ import annotations
from typing import Any, Dict, List

import torch

from .pre_encoder import PreEncoder
from ..col_encoder import EmbeddingEncoder, LinearEncoder
from rllm.types import ColType


class TromptPreEncoder(PreEncoder):
    r"""
    The TromptPreEncoder class is a specialized pre-encoder for the
    Trompt model. It initializes a column-specific encoder dict for
    categorical and numerical features based on the provided metadata.
    Specifically, it uses `EmbeddingEncoder` for categorical features and
    `LinearEncoder` for numerical features.

    Args:
        out_dim (int): The output dimensionality.
        metadata (Dict[rllm.types.ColType, List[Dict[str, Any]]]):
            Metadata for each column type, specifying the statistics and
            properties of the columns.
        in_dim (int, optional): The input dimensionality for numerical features
            (default: :obj:`1`).

    Returns:
        This class does not return a tensor in ``__init__``.
        Encoded outputs are produced by inherited ``forward``.
    """

    def __init__(
        self,
        out_dim: int,
        metadata: Dict[ColType, List[Dict[str, Any]]],
        in_dim: int = 1,
    ) -> None:
        col_encoder_dict = {
            ColType.CATEGORICAL: EmbeddingEncoder(
                post_module=torch.nn.LayerNorm(out_dim)
            ),
            ColType.NUMERICAL: LinearEncoder(
                in_dim=in_dim,
                post_module=torch.nn.ModuleList(
                    [torch.nn.ReLU(), torch.nn.LayerNorm(out_dim)]
                ),
            ),
        }
        super().__init__(out_dim, metadata, col_encoder_dict)
