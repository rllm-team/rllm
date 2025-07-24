from __future__ import annotations
from typing import Any, Dict, List

import torch
from torch import Tensor

from ._col_encoder import ColEncoder, _reset_parameters_soft
from rllm.types import ColType, StatType


class TransTabWordEmbeddingEncoder(ColEncoder):
    r"""Word embedding encoder for categorical and binary features, matching
    the original TransTabWordEmbedding implementation.

    This encoder maps input token indices to dense embeddings, applies
    LayerNorm and Dropout, and optionally a post-processing module.

    Args:
        vocab_size (int): Size of the vocabulary (including padding token).
        out_dim (int): Dimensionality of the output embeddings.
        padding_idx (int): Index in the vocabulary to use as padding.
            (default: 0)
        hidden_dropout_prob (float): Dropout probability after embedding.
            (default: 0.0)
        layer_norm_eps (float): Epsilon value for numerical stability in
            LayerNorm. (default: 1e-5)
        stats_list (Optional[List[Dict[StatType, Any]]]): Precomputed column
            statistics for normalizing or scaling (unused by this encoder).
            (default: None)
        post_module (Optional[torch.nn.Module]): Optional module to apply
            after the core encoding (e.g., additional normalization or
            activation). (default: None)
    """

    supported_types = {ColType.CATEGORICAL, ColType.BINARY}

    def __init__(
        self,
        vocab_size: int,
        out_dim: int,
        padding_idx: int = 0,
        hidden_dropout_prob: float = 0.0,
        layer_norm_eps: float = 1e-5,
        stats_list: List[Dict[StatType, Any]] | None = None,
        post_module: torch.nn.Module | None = None,
    ) -> None:
        super().__init__(out_dim, stats_list, post_module)
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.out_dim = out_dim
        self.hidden_dropout_prob = hidden_dropout_prob
        self.layer_norm_eps = layer_norm_eps

        # Initialize modules exactly as original TransTabWordEmbedding
        self.word_embeddings = torch.nn.Embedding(
            self.vocab_size, self.out_dim, self.padding_idx
        )
        torch.nn.init.kaiming_normal_(self.word_embeddings.weight)
        self.norm = torch.nn.LayerNorm(self.out_dim, eps=self.layer_norm_eps)
        self.dropout = torch.nn.Dropout(self.hidden_dropout_prob)

    def reset_parameters(self) -> None:
        # Reset embedding parameters
        self.word_embeddings.reset_parameters()
        # Reset norm and dropout if they support reset
        _reset_parameters_soft(self.norm)
        _reset_parameters_soft(self.dropout)
        # Reset any post_module
        super().reset_parameters()

    def encode_forward(
        self,
        feat: Tensor,
        col_names: List[str] | None = None,
    ) -> Tensor:
        """
        The core embedding logic is exactly the same as the original forward:
        lookup -> LayerNorm -> Dropout
        """
        x = self.word_embeddings(feat)
        x = self.norm(x)
        x = self.dropout(x)
        return x

    def post_init(self):
        """
        Post-initialization hook required by ColEncoder.
        Currently no additional operation is required
        """
        return

    def forward(
        self,
        feat: Tensor,
        col_names: List[str] | None = None,
    ) -> Tensor:
        # Directly mirror original TransTabWordEmbedding.forward
        # lookup -> norm -> dropout
        x = self.word_embeddings(feat)
        x = self.norm(x)
        x = self.dropout(x)
        return x
