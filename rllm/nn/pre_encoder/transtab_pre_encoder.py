from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
import os

import pandas as pd
import torch
from torch import Tensor

from .pre_encoder import PreEncoder
from ._transtab_word_embedding_encoder import TransTabWordEmbeddingEncoder
from ._transtab_num_embedding_encoder import TransTabNumEmbeddingEncoder
from rllm.types import ColType
from rllm.preprocessing import TransTabDataExtractor
from rllm.data.table_data import TableData


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
        extractor: Optional[TransTabDataExtractor] = None,
        use_align_layer: bool = True,
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
            ColType.NUMERICAL: TransTabNumEmbeddingEncoder(hidden_dim=out_dim),
        }
        super().__init__(out_dim, metadata, col_pre_encoder_dict)
        self.extractor = (
            extractor
            if extractor is not None
            else TransTabDataExtractor(
                categorical_columns=None,
                numerical_columns=None,
                binary_columns=None,
            )
        )
        self.align_layer = (
            torch.nn.Linear(out_dim, out_dim, bias=False)
            if use_align_layer
            else torch.nn.Identity()
        )

    @property
    def device(self) -> torch.device:
        """Dynamically get the device of model parameters."""
        return next(self.parameters()).device

    def _encode_feat_dict(
        self,
        feat_dict: Dict[ColType, Tensor | Tuple[Tensor, ...]],
    ) -> Dict[ColType, Tensor]:
        feat_encoded: Dict[ColType, Tensor] = {}

        for col_type, feat in feat_dict.items():
            if col_type == ColType.NUMERICAL:
                col_ids, col_mask, raw_vals = (
                    feat  # [n_cols, L], [n_cols, L], [B, n_cols]
                )
                token_emb = self.pre_encoder_dict[ColType.CATEGORICAL.value](
                    col_ids
                )  # [n_cols, L, D]
                mask = col_mask.unsqueeze(-1)  # [n_cols, L, 1]
                token_emb = token_emb * mask
                col_emb = token_emb.sum(1) / mask.sum(1)  # [n_cols, D]
                num_emb = self.pre_encoder_dict[ColType.NUMERICAL.value](
                    col_emb, raw_vals=raw_vals
                )
                feat_encoded[col_type] = num_emb  # [B, n_num, D]
            else:
                if isinstance(feat, tuple):
                    input_ids = feat[0]
                else:
                    input_ids = feat
                feat_encoded[col_type] = self.pre_encoder_dict[col_type.value](
                    input_ids
                )

        return feat_encoded

    def _collect_masks_from_inputs(
        self,
        feat_dict: Dict[ColType, Tensor | Tuple[Tensor, ...]],
        emb_dict: Dict[ColType, Tensor],
        df_masks: Optional[Dict[str, Tensor]],
    ) -> Dict[ColType, Tensor]:
        masks: Dict[ColType, Tensor] = {}

        # Numerical: ones mask
        if ColType.NUMERICAL in emb_dict:
            B, n_num, _ = emb_dict[ColType.NUMERICAL].shape
            masks[ColType.NUMERICAL] = torch.ones(B, n_num, device=self.device)

        # From DataFrame path
        if df_masks is not None:
            if "cat_att_mask" in df_masks and ColType.CATEGORICAL in emb_dict:
                masks[ColType.CATEGORICAL] = (
                    df_masks["cat_att_mask"].to(self.device).float()
                )
            if "bin_att_mask" in df_masks and ColType.BINARY in emb_dict:
                masks[ColType.BINARY] = df_masks["bin_att_mask"].to(self.device).float()

        # From feat_dict path (optional tuple masks)
        else:
            for ct in (ColType.CATEGORICAL, ColType.BINARY):
                if ct in emb_dict:
                    feat = feat_dict.get(ct)
                    if isinstance(feat, tuple) and len(feat) >= 2:
                        masks[ct] = feat[1].to(self.device).float()  # provided att_mask
                    else:
                        B, n_cols, _ = emb_dict[ct].shape
                        masks[ct] = torch.ones(B, n_cols, device=self.device)

        return masks

    def _align_and_concat(
        self,
        emb_dict: Dict[ColType, Tensor],
        masks: Dict[ColType, Tensor],
    ) -> Dict[str, Tensor]:
        emb_list: List[Tensor] = []
        mask_list: List[Tensor] = []

        # Numerical
        if ColType.NUMERICAL in emb_dict:
            num_emb = self.align_layer(emb_dict[ColType.NUMERICAL])
            emb_list.append(num_emb)
            mask_list.append(masks[ColType.NUMERICAL])

        # Categorical
        if ColType.CATEGORICAL in emb_dict:
            cat_emb = self.align_layer(emb_dict[ColType.CATEGORICAL])
            emb_list.append(cat_emb)
            mask_list.append(masks[ColType.CATEGORICAL])

        # Binary
        if ColType.BINARY in emb_dict:
            bin_emb = self.align_layer(emb_dict[ColType.BINARY])
            emb_list.append(bin_emb)
            mask_list.append(masks[ColType.BINARY])

        if len(emb_list) == 0:
            raise ValueError(
                "No features were encoded; check extractor/columns configuration."
            )

        all_emb = torch.cat(emb_list, dim=1)  # [B, total_seq_len, D]
        all_mask = torch.cat(mask_list, dim=1)  # [B, total_seq_len]
        return {"embedding": all_emb, "attention_mask": all_mask}

    def forward(
        self,
        x: Union[pd.DataFrame, Dict[ColType, Tensor | Tuple[Tensor, ...]], "TableData"],
        *,
        shuffle: bool = False,
        align_and_concat: bool = True,
        return_dict: bool = False,
        requires_grad: bool = False,
    ) -> Union[Dict[str, Tensor], Dict[ColType, Tensor], Tensor]:
        grad_ctx = (
            (lambda: torch.enable_grad())
            if requires_grad
            else (lambda: torch.no_grad())
        )
        with grad_ctx():
            # Check if x is a TableData object or TableData-like object
            if isinstance(x, TableData) or hasattr(x, "feat_dict"):
                # Extract feat_dict and colname_token_ids from TableData
                if (
                    hasattr(x, "if_materialized")
                    and callable(x.if_materialized)
                    and not x.if_materialized()
                ):
                    raise ValueError(
                        "TableData must be materialized before passing to TransTabPreEncoder. "
                        "Call table_data.lazy_materialize() first."
                    )

                # Use extractor to convert feat_dict to TransTab format
                data = self.extractor(
                    shuffle=shuffle,
                    feat_dict=x.feat_dict,
                    colname_token_ids=getattr(x, "colname_token_ids", None),
                )

                feat_dict: Dict[ColType, Tensor | Tuple[Tensor, ...]] = {}
                if data["x_cat_input_ids"] is not None:
                    feat_dict[ColType.CATEGORICAL] = (
                        data["x_cat_input_ids"].to(self.device),
                        data["cat_att_mask"].to(self.device),
                    )
                if data["x_bin_input_ids"] is not None:
                    feat_dict[ColType.BINARY] = (
                        data["x_bin_input_ids"].to(self.device),
                        data["bin_att_mask"].to(self.device),
                    )
                if data["x_num"] is not None:
                    feat_dict[ColType.NUMERICAL] = (
                        data["num_col_input_ids"].to(self.device),
                        data["num_att_mask"].to(self.device),
                        data["x_num"].to(self.device),
                    )

                emb_dict = self._encode_feat_dict(feat_dict)
                if not align_and_concat:
                    if return_dict:
                        return emb_dict
                    return (
                        torch.cat(list(emb_dict.values()), dim=1)
                        if len(emb_dict) > 0
                        else None
                    )

                df_masks = {
                    "cat_att_mask": (
                        data["cat_att_mask"]
                        if data["cat_att_mask"] is not None
                        else None
                    ),
                    "bin_att_mask": (
                        data["bin_att_mask"]
                        if data["bin_att_mask"] is not None
                        else None
                    ),
                }
                masks = self._collect_masks_from_inputs(
                    feat_dict, emb_dict, df_masks=df_masks
                )
                return self._align_and_concat(emb_dict, masks)
            else:
                raise TypeError(
                    "TransTabPreEncoder.forward: x must be a pandas.DataFrame or a feat_dict mapping."
                )

    def save(self, path: str) -> None:
        self.extractor.save(path)
        os.makedirs(path, exist_ok=True)
        encoder_path = os.path.join(path, "input_encoder.bin")
        torch.save(self.state_dict(), encoder_path)
        print(f"Saved pre_encoder (integrated) weights to {encoder_path}")

    def load(self, ckpt_dir: str) -> None:
        self.extractor.load(ckpt_dir)
        encoder_path = os.path.join(ckpt_dir, "input_encoder.bin")
        try:
            state_dict = torch.load(encoder_path, map_location='cpu', weights_only=True)
        except TypeError:
            state_dict = torch.load(encoder_path, map_location='cpu')
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"Loaded pre_encoder (integrated) weights from {encoder_path}")
        print(f" Missing keys: {missing}")
        print(f" Unexpected keys: {unexpected}")
