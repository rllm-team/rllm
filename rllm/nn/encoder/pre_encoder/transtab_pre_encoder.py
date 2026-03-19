from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
import collections
import json
import os

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from transformers import BertTokenizerFast

from .pre_encoder import PreEncoder
from ..col_encoder import (
    TransTabNumEmbeddingEncoder,
    TransTabWordEmbeddingEncoder,
)
from rllm.types import ColType
from rllm.data.table_data import TableData


class TransTabPreEncoder(PreEncoder):
    r"""Pre-encoder for the TransTab model as proposed in
    `"TransTab: Learning Transferable Tabular Transformers Across Tables"
    <https://arxiv.org/abs/2205.09328>`_ paper.

    This module integrates tokenizer management, column-type bookkeeping,
    and feature extraction into a single pre-encoder that converts a
    :class:`~rllm.data.table_data.TableData` ``feat_dict`` into embeddings
    consumable by downstream Transformer layers.

    Specifically it:

    1. Manages a :class:`BertTokenizerFast` instance used to tokenize
       categorical (as text) and binary column names at inference time.
    2. Maintains ordered lists of *categorical*, *numerical*, and *binary*
       column names, with duplicate detection and optional auto-resolution.
    3. Adapts a ``feat_dict`` produced by :class:`TableData` into the
       TransTab-specific tensor layout (categorical token IDs, numerical
       column-name token IDs + raw values, binary per-row token IDs).
    4. Encodes the adapted tensors via word-embedding and numeric-embedding
       sub-encoders, then optionally aligns and concatenates them.

    Args:
        out_dim (int): Output embedding dimensionality (d_model).
        metadata (Dict[ColType, List[Dict[str, Any]]]): Metadata mapping from
            column type to list of per-column statistics dicts.
        categorical_columns (Optional[List[str]]): Names of categorical columns.
        numerical_columns (Optional[List[str]]): Names of numerical columns.
        binary_columns (Optional[List[str]]): Names of binary (0/1) columns.
        tokenizer (Optional[BertTokenizerFast]): A pre-initialized tokenizer.
            Takes precedence over *tokenizer_dir* when provided.
        tokenizer_dir (str): Path to a pretrained tokenizer directory.  If
            *tokenizer* is ``None`` and the directory does not exist,
            ``"bert-base-uncased"`` is downloaded and saved here.
        hidden_dropout_prob (float): Dropout probability for the word-embedding
            sub-encoder.
        layer_norm_eps (float): Epsilon for LayerNorm in the word-embedding
            sub-encoder.
        use_align_layer (bool): If ``True``, apply a linear alignment projection
            before concatenation.
        disable_tokenizer_parallel (bool): If ``True``, set the environment
            variable ``TOKENIZERS_PARALLELISM=false``.
        ignore_duplicate_cols (bool): If ``True``, automatically rename
            duplicate column names; otherwise raise :class:`ValueError`.

    Returns:
        This class does not return tensors in ``__init__``.
        The ``forward`` method returns either aligned embeddings with attention
        masks, a dictionary of encoded tensors, or a concatenated tensor.
    """

    def __init__(
        self,
        out_dim: int,
        metadata: Dict[ColType, List[Dict[str, Any]]],
        categorical_columns: Optional[List[str]] = None,
        numerical_columns: Optional[List[str]] = None,
        binary_columns: Optional[List[str]] = None,
        tokenizer: Optional[BertTokenizerFast] = None,
        tokenizer_dir: str = "./tokenizer",
        hidden_dropout_prob: float = 0.0,
        layer_norm_eps: float = 1e-5,
        use_align_layer: bool = True,
        disable_tokenizer_parallel: bool = True,
        ignore_duplicate_cols: bool = False,
    ) -> None:
        self._init_tokenizer(tokenizer, tokenizer_dir, disable_tokenizer_parallel)
        self._init_columns(
            categorical_columns,
            numerical_columns,
            binary_columns,
            ignore_duplicate_cols,
        )

        col_encoder_dict = {
            ColType.CATEGORICAL: TransTabWordEmbeddingEncoder(
                vocab_size=self.tokenizer.vocab_size,
                out_dim=out_dim,
                padding_idx=self.tokenizer.pad_token_id,
                hidden_dropout_prob=hidden_dropout_prob,
                layer_norm_eps=layer_norm_eps,
            ),
            ColType.BINARY: TransTabWordEmbeddingEncoder(
                vocab_size=self.tokenizer.vocab_size,
                out_dim=out_dim,
                padding_idx=self.tokenizer.pad_token_id,
                hidden_dropout_prob=hidden_dropout_prob,
                layer_norm_eps=layer_norm_eps,
            ),
            ColType.NUMERICAL: TransTabNumEmbeddingEncoder(hidden_dim=out_dim),
        }
        super().__init__(out_dim, metadata, col_encoder_dict)

        self.align_layer = (
            torch.nn.Linear(out_dim, out_dim, bias=False)
            if use_align_layer
            else torch.nn.Identity()
        )

    # Tokenizer management
    def _init_tokenizer(
        self,
        tokenizer: Optional[BertTokenizerFast],
        tokenizer_dir: str,
        disable_tokenizer_parallel: bool,
    ) -> None:
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif os.path.exists(tokenizer_dir):
            self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_dir)
        else:
            self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
            self.tokenizer.save_pretrained(tokenizer_dir)
        self.tokenizer.model_max_length = 512
        if disable_tokenizer_parallel:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Column-name management
    @staticmethod
    def _deduplicate_preserve_order(seq: List[str]) -> List[str]:
        seen: set[str] = set()
        result: List[str] = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                result.append(x)
        return result

    def _init_columns(
        self,
        categorical_columns: Optional[List[str]],
        numerical_columns: Optional[List[str]],
        binary_columns: Optional[List[str]],
        ignore_duplicate_cols: bool,
    ) -> None:
        self.categorical_columns: List[str] = (
            self._deduplicate_preserve_order(categorical_columns)
            if categorical_columns
            else []
        )
        self.numerical_columns: List[str] = (
            self._deduplicate_preserve_order(numerical_columns)
            if numerical_columns
            else []
        )
        self.binary_columns: List[str] = (
            self._deduplicate_preserve_order(binary_columns) if binary_columns else []
        )
        self.ignore_duplicate_cols = ignore_duplicate_cols

        col_ok, dup = self._check_column_overlap(
            self.categorical_columns, self.numerical_columns, self.binary_columns
        )
        if not col_ok:
            if not self.ignore_duplicate_cols:
                for c in dup:
                    print(
                        f"ERROR: Find duplicate cols named `{c}`; "
                        f"set ignore_duplicate_cols=True to auto-resolve."
                    )
                raise ValueError("Column overlap detected; aborting.")
            else:
                self._solve_duplicate_cols(dup)

    def update(
        self,
        cat: Optional[List[str]] = None,
        num: Optional[List[str]] = None,
        bin: Optional[List[str]] = None,
    ) -> None:
        r"""Dynamically extend column lists and recheck for duplicates.

        Args:
            cat: New categorical columns to add.
            num: New numerical columns to add.
            bin: New binary columns to add.

        Raises:
            ValueError: If duplicate columns are detected after the update and
                *ignore_duplicate_cols* is ``False``.
        """
        if cat:
            self.categorical_columns.extend(cat)
            self.categorical_columns = list(set(self.categorical_columns))
        if num:
            self.numerical_columns.extend(num)
            self.numerical_columns = list(set(self.numerical_columns))
        if bin:
            self.binary_columns.extend(bin)
            self.binary_columns = list(set(self.binary_columns))

        col_ok, dup = self._check_column_overlap(
            self.categorical_columns, self.numerical_columns, self.binary_columns
        )
        if not col_ok:
            if not self.ignore_duplicate_cols:
                for c in dup:
                    print(
                        f"ERROR: Find duplicate cols named `{c}`; "
                        f"set ignore_duplicate_cols=True to auto-resolve."
                    )
                raise ValueError("Column overlap detected after update; aborting.")
            else:
                self._solve_duplicate_cols(dup)

    @staticmethod
    def _check_column_overlap(
        cat_cols: Optional[List[str]] = None,
        num_cols: Optional[List[str]] = None,
        bin_cols: Optional[List[str]] = None,
    ) -> Tuple[bool, List[str]]:
        all_cols: List[str] = []
        if cat_cols:
            all_cols += cat_cols
        if num_cols:
            all_cols += num_cols
        if bin_cols:
            all_cols += bin_cols
        if not all_cols:
            print("WARNING: No columns specified; default to categorical.")
            return True, []
        counter = collections.Counter(all_cols)
        dup = [col for col, cnt in counter.items() if cnt > 1]
        return len(dup) == 0, dup

    def _solve_duplicate_cols(self, duplicate_cols: List[str]) -> None:
        for col in duplicate_cols:
            print(f"WARNING: Auto-resolving duplicate column `{col}`")
            if col in self.categorical_columns:
                self.categorical_columns.remove(col)
                self.categorical_columns.append(f"[cat]{col}")
            if col in self.numerical_columns:
                self.numerical_columns.remove(col)
                self.numerical_columns.append(f"[num]{col}")
            if col in self.binary_columns:
                self.binary_columns.remove(col)
                self.binary_columns.append(f"[bin]{col}")

    # feat_dict adaptation (convert TableData feat_dict → TransTab layout)
    def _adapt_feat_dict(
        self,
        feat_dict: Dict[ColType, Tensor | Tuple[Tensor, Tensor]],
        colname_token_ids: Optional[Dict[str, Tuple[Tensor, Tensor]]] = None,
        shuffle: bool = False,
    ) -> Dict[str, Tensor | None]:
        r"""Adapt a pre-computed ``feat_dict`` from :class:`TableData` into the
        tensor layout expected by TransTab sub-encoders.

        Args:
            feat_dict: Feature dictionary from :class:`TableData`:
                ``ColType.TEXT`` → ``(input_ids [B, L], attention_mask [B, L])``,
                ``ColType.NUMERICAL`` → ``[B, n_num]``,
                ``ColType.BINARY`` → ``[B, n_bin]``.
            colname_token_ids: Mapping from column name to
                ``(token_ids, attention_mask)`` tensors.
            shuffle: If ``True``, randomly shuffle column order within each
                type.

        Returns:
            Dict with keys ``x_num``, ``num_col_input_ids``, ``num_att_mask``,
            ``x_cat_input_ids``, ``cat_att_mask``, ``x_bin_input_ids``,
            ``bin_att_mask``; values are tensors or ``None``.
        """
        out: Dict[str, Tensor | None] = {
            "x_num": None,
            "num_col_input_ids": None,
            "num_att_mask": None,
            "x_cat_input_ids": None,
            "cat_att_mask": None,
            "x_bin_input_ids": None,
            "bin_att_mask": None,
        }

        # TEXT (mapped to TransTab categorical)
        if ColType.TEXT in feat_dict:
            text_data = feat_dict[ColType.TEXT]
            if isinstance(text_data, tuple):
                out["x_cat_input_ids"] = text_data[0].long()
                out["cat_att_mask"] = text_data[1].long()
            else:
                raise ValueError(
                    "TEXT features must be tokenized (tuple of ids and mask)"
                )

        if ColType.NUMERICAL in feat_dict:
            out["x_num"] = feat_dict[ColType.NUMERICAL].float()
            if colname_token_ids is not None:
                num_cols = [
                    c for c in colname_token_ids.keys() if c in self.numerical_columns
                ]
                if shuffle:
                    np.random.shuffle(num_cols)
                if num_cols:
                    num_ids_list = [colname_token_ids[c][0] for c in num_cols]
                    num_mask_list = [colname_token_ids[c][1] for c in num_cols]
                    out["num_col_input_ids"] = torch.stack(num_ids_list, dim=0).long()
                    out["num_att_mask"] = torch.stack(num_mask_list, dim=0).long()

        if ColType.BINARY in feat_dict:
            x_bin = feat_dict[ColType.BINARY]
            if colname_token_ids is not None:
                bin_cols = [
                    c for c in colname_token_ids.keys() if c in self.binary_columns
                ]
                if shuffle:
                    np.random.shuffle(bin_cols)
                if bin_cols:
                    batch_size = x_bin.shape[0]
                    bin_texts: List[str] = []
                    for i in range(batch_size):
                        active_cols = [
                            col
                            for j, col in enumerate(bin_cols)
                            if j < x_bin.shape[1] and x_bin[i, j].item() > 0.5
                        ]
                        bin_texts.append(" ".join(active_cols))
                    tokens = self.tokenizer(
                        bin_texts,
                        padding=True,
                        truncation=True,
                        add_special_tokens=False,
                        return_tensors="pt",
                    )
                    if tokens["input_ids"].shape[1] > 0:
                        out["x_bin_input_ids"] = tokens["input_ids"]
                        out["bin_att_mask"] = tokens["attention_mask"]

        return out

    # Encoding helpers
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _encode_feat_dict(
        self,
        feat_dict: Dict[ColType, Tensor | Tuple[Tensor, ...]],
    ) -> Dict[ColType, Tensor]:
        feat_encoded: Dict[ColType, Tensor] = {}
        for col_type, feat in feat_dict.items():
            if col_type == ColType.NUMERICAL:
                col_ids, col_mask, raw_vals = feat
                token_emb = self.col_encoder_dict[ColType.CATEGORICAL.value](col_ids)
                mask = col_mask.unsqueeze(-1)
                token_emb = token_emb * mask
                col_emb = token_emb.sum(1) / mask.sum(1)
                num_emb = self.col_encoder_dict[ColType.NUMERICAL.value](
                    col_emb, raw_vals=raw_vals
                )
                feat_encoded[col_type] = num_emb
            else:
                if isinstance(feat, tuple):
                    input_ids = feat[0]
                else:
                    input_ids = feat
                feat_encoded[col_type] = self.col_encoder_dict[col_type.value](
                    input_ids
                )
        return feat_encoded

    def _collect_masks(
        self,
        feat_dict: Dict[ColType, Tensor | Tuple[Tensor, ...]],
        emb_dict: Dict[ColType, Tensor],
        df_masks: Optional[Dict[str, Tensor]],
    ) -> Dict[ColType, Tensor]:
        masks: Dict[ColType, Tensor] = {}

        if ColType.NUMERICAL in emb_dict:
            B, n_num, _ = emb_dict[ColType.NUMERICAL].shape
            masks[ColType.NUMERICAL] = torch.ones(B, n_num, device=self.device)

        if df_masks is not None:
            if "cat_att_mask" in df_masks and ColType.CATEGORICAL in emb_dict:
                masks[ColType.CATEGORICAL] = (
                    df_masks["cat_att_mask"].to(self.device).float()
                )
            if "bin_att_mask" in df_masks and ColType.BINARY in emb_dict:
                masks[ColType.BINARY] = df_masks["bin_att_mask"].to(self.device).float()
        else:
            for ct in (ColType.CATEGORICAL, ColType.BINARY):
                if ct in emb_dict:
                    feat = feat_dict.get(ct)
                    if isinstance(feat, tuple) and len(feat) >= 2:
                        masks[ct] = feat[1].to(self.device).float()
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

        if ColType.NUMERICAL in emb_dict:
            emb_list.append(self.align_layer(emb_dict[ColType.NUMERICAL]))
            mask_list.append(masks[ColType.NUMERICAL])
        if ColType.CATEGORICAL in emb_dict:
            emb_list.append(self.align_layer(emb_dict[ColType.CATEGORICAL]))
            mask_list.append(masks[ColType.CATEGORICAL])
        if ColType.BINARY in emb_dict:
            emb_list.append(self.align_layer(emb_dict[ColType.BINARY]))
            mask_list.append(masks[ColType.BINARY])

        if len(emb_list) == 0:
            raise ValueError("No features were encoded; check column configuration.")

        all_emb = torch.cat(emb_list, dim=1)  # [B, total_seq_len, D]
        all_mask = torch.cat(mask_list, dim=1)  # [B, total_seq_len]
        return {"embedding": all_emb, "attention_mask": all_mask}

    def forward(
        self,
        x: Union[pd.DataFrame, Dict[ColType, Tensor | Tuple[Tensor, ...]], TableData],
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
            if isinstance(x, TableData) or hasattr(x, "feat_dict"):
                if (
                    hasattr(x, "if_materialized")
                    and callable(x.if_materialized)
                    and not x.if_materialized()
                ):
                    raise ValueError(
                        "TableData must be materialized before passing to "
                        "TransTabPreEncoder. Call table_data.lazy_materialize() first."
                    )

                data = self._adapt_feat_dict(
                    feat_dict=x.feat_dict,
                    colname_token_ids=getattr(x, "colname_token_ids", None),
                    shuffle=shuffle,
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
                    "cat_att_mask": data["cat_att_mask"],
                    "bin_att_mask": data["bin_att_mask"],
                }
                masks = self._collect_masks(feat_dict, emb_dict, df_masks=df_masks)
                return self._align_and_concat(emb_dict, masks)
            else:
                raise TypeError(
                    "TransTabPreEncoder.forward: x must be a TableData or an "
                    "object with a feat_dict attribute."
                )

    def save(self, path: str) -> None:
        r"""Save tokenizer, column configuration, and encoder weights.

        On-disk layout::

            {path}/extractor/tokenizer/   # tokenizer files
            {path}/extractor/extractor.json  # column lists
            {path}/input_encoder.bin       # encoder state_dict

        Args:
            path: Base directory.
        """
        # Tokenizer & column config (backward-compatible directory layout)
        save_path = os.path.join(path, "extractor")
        os.makedirs(save_path, exist_ok=True)
        self.tokenizer.save_pretrained(os.path.join(save_path, "tokenizer"))
        col_type_dict = {
            "categorical": self.categorical_columns,
            "numerical": self.numerical_columns,
            "binary": self.binary_columns,
        }
        with open(
            os.path.join(save_path, "extractor.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(col_type_dict, f, ensure_ascii=False)

        # Encoder weights
        os.makedirs(path, exist_ok=True)
        encoder_path = os.path.join(path, "input_encoder.bin")
        torch.save(self.state_dict(), encoder_path)
        print(f"Saved TransTabPreEncoder weights to {encoder_path}")

    def load(self, ckpt_dir: str) -> None:
        r"""Load tokenizer, column configuration, and encoder weights.

        Args:
            ckpt_dir: Directory previously written by :meth:`save`.
        """
        tokenizer_path = os.path.join(ckpt_dir, "extractor", "tokenizer")
        coltype_path = os.path.join(ckpt_dir, "extractor", "extractor.json")

        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
        with open(coltype_path, "r", encoding="utf-8") as f:
            col_type_dict = json.load(f)
        self.categorical_columns = col_type_dict.get("categorical", [])
        self.numerical_columns = col_type_dict.get("numerical", [])
        self.binary_columns = col_type_dict.get("binary", [])
        print(f"Loaded column configuration from {coltype_path}")

        encoder_path = os.path.join(ckpt_dir, "input_encoder.bin")
        try:
            state_dict = torch.load(encoder_path, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(encoder_path, map_location="cpu")
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"Loaded TransTabPreEncoder weights from {encoder_path}")
        print(f"  Missing keys: {missing}")
        print(f"  Unexpected keys: {unexpected}")
