from __future__ import annotations
from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor

from .pre_encoder import PreEncoder
from ._transtab_word_embedding_encoder import TransTabWordEmbeddingEncoder
from ._transtab_num_embedding_encoder import TransTabNumEmbeddingEncoder
from rllm.types import ColType


class TransTabPreEncoder(PreEncoder):
    """
    A specialized PreEncoder for the TransTab model.
    Uses word-based embedding for categorical and binary features,
    and numeric embedding for numerical features.

    Args:
        out_dim: Output embedding dimension for all features.
        metadata: Mapping from ColType to list of column statistics dicts.
        vocab_size: Vocabulary size for token embeddings.
        padding_idx: Padding index for token embeddings.
        hidden_dropout_prob: Dropout probability in token embeddings.
        layer_norm_eps: Epsilon for LayerNorm in token embeddings.
    """

    def __init__(
        self,
        out_dim: int,
        metadata: Dict[ColType, List[Dict[str, Any]]],
        vocab_size: int,
        padding_idx: int = 0,
        hidden_dropout_prob: float = 0.0,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        # Build column-specific encoder mapping
        col_pre_encoder_dict = {
            ColType.CATEGORICAL: TransTabWordEmbeddingEncoder(
                vocab_size=vocab_size,
                out_dim=out_dim,
                padding_idx=padding_idx,
                hidden_dropout_prob=hidden_dropout_prob,
                layer_norm_eps=layer_norm_eps,
            ),
            ColType.BINARY: TransTabWordEmbeddingEncoder(
                vocab_size=vocab_size,
                out_dim=out_dim,
                padding_idx=padding_idx,
                hidden_dropout_prob=hidden_dropout_prob,
                layer_norm_eps=layer_norm_eps,
            ),
            # Only hidden_dim is needed for numeric encoder
            ColType.NUMERICAL: TransTabNumEmbeddingEncoder(
                hidden_dim=out_dim
            ),
        }
        super().__init__(out_dim, metadata, col_pre_encoder_dict)

    def forward(
        self,
        feat_dict: Dict[ColType, Tensor | Tuple[Tensor, Tensor, Tensor]],
        return_dict: bool = False,
    ) -> Any:
        feat_encoded: Dict[ColType, Tensor] = {}

        for col_type, feat in feat_dict.items():
            if col_type == ColType.NUMERICAL:
                # Unpack tuple:
                #   col_ids:   [n_cols, seq_len]
                #   col_mask:  [n_cols, seq_len]
                #   raw_vals:  [batch_size, n_cols]
                col_ids, col_mask, raw_vals = feat

                # 1) Column name token embedding
                token_emb = self.pre_encoder_dict[ColType.CATEGORICAL.value](col_ids)

                # 2) Masked average pooling over token dimension
                mask = col_mask.unsqueeze(-1)            # [n_cols, seq_len, 1]
                token_emb = token_emb * mask             # apply mask
                col_emb = token_emb.sum(1) / mask.sum(1) # [n_cols, out_dim]

                # 3) Scale embeddings by numeric values and add bias
                num_emb = self.pre_encoder_dict[ColType.NUMERICAL.value](
                    col_emb, raw_vals
                )
                feat_encoded[col_type] = num_emb

            else:
                # Categorical / Binary: use original encoder
                feat_encoded[col_type] = self.pre_encoder_dict[col_type.value](feat)  

        if return_dict:
            return feat_encoded

        # Concatenate all feature embeddings along column dimension
        return torch.cat(list(feat_encoded.values()), dim=1)


