from __future__ import annotations
from typing import Dict, List, Any

import torch

from rllm.types import ColType
from rllm.nn.conv.utils import Transformer
from rllm.nn.pre_encoder import TabTransformerPreEncoder


class TabTransformerConv(torch.nn.Module):
    r"""The TabTransformer LayerConv introduced in the
    `"TabTransformer: Tabular Data Modeling Using Contextual Embeddings"
    <https://arxiv.org/abs/2012.06678>`_ paper.

    Args:
        dim (int): Input/output channel dimensionality.
        heads (int): Number of attention heads.
        head_dim (int):  Dimensionality of each attention head.
        attn_dropout (float): attention module dropout (default: :obj:`0.`).
        ffn_dropout (float): attention module dropout (default: :obj:`0.`).
    """

    def __init__(
        self,
        dim,
        heads: int = 8,
        head_dim: int = 16,
        attn_dropout: float = 0.3,
        ff_dropout: float = 0.3,
        metadata: Dict[ColType, List[Dict[str, Any]]] = None,
    ):
        super().__init__()
        self.transformer = Transformer(
            in_dim=dim,
            heads=heads,
            head_dim=head_dim,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
        )
        self.pre_encoder = None
        if metadata:
            self.pre_encoder = TabTransformerPreEncoder(
                out_dim=dim,
                metadata=metadata,
            )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.transformer.reset_parameters()
        if self.pre_encoder is not None:
            self.pre_encoder.reset_parameters()

    def forward(self, x):
        if self.pre_encoder is not None:
            x = self.pre_encoder(x, return_dict=True)

        x[ColType.CATEGORICAL] = self.transformer(x[ColType.CATEGORICAL])

        return x
