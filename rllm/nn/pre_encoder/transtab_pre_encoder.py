from __future__ import annotations
from typing import Any, Dict, List, Tuple, Union
import collections
import json
import os
import logging

import torch
from torch import Tensor
import numpy as np
import pandas as pd
from transformers import BertTokenizerFast

from .pre_encoder import PreEncoder
from ._transtab_word_embedding_encoder import TransTabWordEmbeddingEncoder
from ._transtab_num_embedding_encoder import TransTabNumEmbeddingEncoder
from rllm.types import ColType


TRAINING_ARGS_NAME = "training_args.json"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
WEIGHTS_NAME = "pytorch_model.bin"
TOKENIZER_DIR = 'tokenizer'
EXTRACTOR_STATE_DIR = 'extractor'
EXTRACTOR_STATE_NAME = 'extractor.json'
INPUT_ENCODER_NAME = 'input_encoder.bin'

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class TransTabDataExtractor:
    r"""TransTabDataExtractor: Extract raw DataFrame columns into token IDs and value tensors,
    matching original TransTabFeatureExtractor behavior as proposed in
    `"TransTab: Learning Transferable Tabular Transformers Across Tables"`
    <https://arxiv.org/abs/2205.09328>`_ paper.

    This class converts columns of a pandas.DataFrame—divided by type
    (categorical, numerical, binary)—into PyTorch tensors and tokenized
    inputs suitable for downstream Transformer models.

    Args:
        categorical_columns (Optional[List[str]]): Names of categorical columns.
            If None, defaults to empty list. (default: None)
        numerical_columns (Optional[List[str]]): Names of numerical columns.
            If None, defaults to empty list. (default: None)
        binary_columns (Optional[List[str]]): Names of binary (0/1) columns.
            If None, defaults to empty list. (default: None)
        tokenizer_dir (str): Path to pretrained tokenizer directory. If it
            does not exist, “bert-base-uncased” is downloaded and saved here.
            (default: "./transtab/tokenizer")
        disable_tokenizer_parallel (bool): If True, sets
            TOKENIZERS_PARALLELISM="false" to disable parallelism. (default: True)
        ignore_duplicate_cols (bool): If True, automatically renames duplicate
            column names; otherwise raises ValueError on duplicates. (default: False)

    Raises:
        ValueError: If duplicate columns are detected and `ignore_duplicate_cols`
            is False.
    """

    def __init__(
        self,
        categorical_columns: list[str] | None = None,
        numerical_columns: list[str] | None = None,
        binary_columns: list[str] | None = None,
        tokenizer_dir: str = "./tokenizer",
        disable_tokenizer_parallel: bool = True,
        ignore_duplicate_cols: bool = False,
    ) -> None:
        # Initialize tokenizer
        if os.path.exists(tokenizer_dir):
            self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_dir)
        else:
            self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
            self.tokenizer.save_pretrained(tokenizer_dir)
        self.tokenizer.model_max_length = 512
        if disable_tokenizer_parallel:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Column grouping
        self.categorical_columns = list(set(categorical_columns)) if categorical_columns else []
        self.numerical_columns = list(set(numerical_columns)) if numerical_columns else []
        self.binary_columns = list(set(binary_columns)) if binary_columns else []
        self.ignore_duplicate_cols = ignore_duplicate_cols

        # Check and handle duplicate column names
        col_ok, dup = self._check_column_overlap(self.categorical_columns,
                                                 self.numerical_columns,
                                                 self.binary_columns)
        if not col_ok:
            if not self.ignore_duplicate_cols:
                for c in dup:
                    logger.error(f'Find duplicate cols named `{c}`; set ignore_duplicate_cols=True to auto-resolve.')
                raise ValueError("Column overlap detected; aborting.")
            else:
                self._solve_duplicate_cols(dup)

    def __call__(
        self,
        df: pd.DataFrame,
        shuffle: bool = False,
    ) -> dict[str, Tensor | None]:
        r"""Convert DataFrame columns into tensors and token sequences.

        Args:
            df (pd.DataFrame): Input data frame containing configured columns.
            shuffle (bool): If True, shuffles the order of columns within each type.
                (default: False)

        Returns:
            Dict[str, Optional[Tensor]] with keys:
              - "x_num": Float tensor of numerical values or None.
              - "num_col_input_ids": Long tensor of input IDs for numerical column names or None.
              - "num_att_mask": Long tensor attention mask for numerical tokens or None.
              - "x_cat_input_ids": Long tensor of input IDs for categorical tokens or None.
              - "cat_att_mask": Long tensor attention mask for categorical tokens or None.
              - "x_bin_input_ids": Long tensor of input IDs for binary tokens or None.
              - "bin_att_mask": Long tensor attention mask for binary tokens or None.
        """
        cols = df.columns.tolist()
        cat_cols = [c for c in cols if self.categorical_columns and c in self.categorical_columns]
        num_cols = [c for c in cols if self.numerical_columns and c in self.numerical_columns]
        bin_cols = [c for c in cols if self.binary_columns and c in self.binary_columns]

        configured = bool(self.categorical_columns or self.numerical_columns or self.binary_columns)
        if not any((cat_cols, num_cols, bin_cols)):
            if configured:
                raise ValueError("Configured cat/num/bin columns, but none matched DataFrame columns.")
            else:
                cat_cols = cols

        if shuffle:
            np.random.shuffle(cat_cols)
            np.random.shuffle(num_cols)
            np.random.shuffle(bin_cols)

        out: dict[str, Tensor | None] = {
            "x_num": None,
            "num_col_input_ids": None,
            "num_att_mask": None,
            "x_cat_input_ids": None,
            "cat_att_mask": None,
            "x_bin_input_ids": None,
            "bin_att_mask": None,
        }

        # Numerical columns
        if num_cols:
            x_num_df = df[num_cols].infer_objects(copy=False).fillna(0)
            out["x_num"] = torch.tensor(x_num_df.values, dtype=torch.float32)
            tokens = self.tokenizer(
                num_cols,
                padding=True,
                truncation=True,
                add_special_tokens=False,
                return_tensors="pt",
            )
            out["num_col_input_ids"] = tokens["input_ids"]
            out["num_att_mask"] = tokens["attention_mask"]

        # Categorical columns
        if cat_cols:
            x_cat = df[cat_cols]
            mask = (~x_cat.isna()).astype(int)
            with pd.option_context('future.no_silent_downcasting', True):
                x_cat = x_cat.fillna("")
            x_cat = x_cat.astype(str)
            cat_texts: list[str] = []
            for values, flags in zip(x_cat.values, mask.values):
                tokens = [f"{col} {val}" for col, val, flag in zip(cat_cols, values, flags) if flag]
                cat_texts.append(" ".join(tokens))
            tokens = self.tokenizer(
                cat_texts,
                padding=True,
                truncation=True,
                add_special_tokens=False,
                return_tensors="pt",
            )
            out["x_cat_input_ids"] = tokens["input_ids"]
            out["cat_att_mask"] = tokens["attention_mask"]

        # Binary columns
        if bin_cols:
            x_bin = df[bin_cols].copy()
            try:
                x_bin = x_bin.astype(str).map(lambda s: s.strip().lower())
            except AttributeError:
                x_bin = x_bin.astype(str).applymap(lambda s: s.strip().lower())

            POS = {"1", "true", "t", "yes", "y", "on"}
            NEG = {"0", "false", "f", "no", "n", "off", "", "nan", "none", "null"}
            pos_mask = x_bin.isin(POS)
            neg_mask = x_bin.isin(NEG)
            rem = x_bin.where(~(pos_mask | neg_mask))
            num = rem.apply(pd.to_numeric, errors="coerce")
            gt0 = (num.fillna(0).astype(float) > 0)
            x_bin_int = (pos_mask | gt0).astype(int)

            bin_texts: list[str] = []
            for values in x_bin_int.values:
                tokens = [col for col, flag in zip(bin_cols, values) if flag]
                bin_texts.append(" ".join(tokens))

            tokens = self.tokenizer(
                bin_texts, padding=True, truncation=True,
                add_special_tokens=False, return_tensors="pt",
            )
            if tokens["input_ids"].shape[1] > 0:
                out["x_bin_input_ids"] = tokens["input_ids"]
                out["bin_att_mask"] = tokens["attention_mask"]

        return out

    def save(self, path: str) -> None:
        r"""Save tokenizer and column grouping configuration to disk.

        The extractor state is saved under:
          {path}/{EXTRACTOR_STATE_DIR}/tokenizer/  # tokenizer files
          {path}/{EXTRACTOR_STATE_DIR}/{EXTRACTOR_STATE_NAME}  # JSON of column lists

        Args:
            path (str): Base directory in which to save extractor state.
        """
        save_path = os.path.join(path, EXTRACTOR_STATE_DIR)
        os.makedirs(save_path, exist_ok=True)

        tokenizer_path = os.path.join(save_path, TOKENIZER_DIR)
        self.tokenizer.save_pretrained(tokenizer_path)

        coltype_path = os.path.join(save_path, EXTRACTOR_STATE_NAME)
        col_type_dict = {
            'categorical': self.categorical_columns,
            'numerical': self.numerical_columns,
            'binary': self.binary_columns,
        }
        with open(coltype_path, 'w', encoding='utf-8') as f:
            json.dump(col_type_dict, f, ensure_ascii=False)

    def load(self, path: str) -> None:
        r"""Load tokenizer and column grouping from disk.

        Args:
            path (str): Base directory containing extractor state.
        """
        tokenizer_path = os.path.join(path, EXTRACTOR_STATE_DIR, TOKENIZER_DIR)
        coltype_path = os.path.join(path, EXTRACTOR_STATE_DIR, EXTRACTOR_STATE_NAME)

        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
        with open(coltype_path, 'r', encoding='utf-8') as f:
            col_type_dict = json.load(f)
        self.categorical_columns = col_type_dict.get('categorical', [])
        self.numerical_columns = col_type_dict.get('numerical', [])
        self.binary_columns = col_type_dict.get('binary', [])
        logger.info(f'Loaded extractor state from {coltype_path}')

    def update(
        self,
        cat: list[str] | None = None,
        num: list[str] | None = None,
        bin: list[str] | None = None,
    ) -> None:
        r"""Dynamically extend column lists and recheck for duplicates.

        Args:
            cat (Optional[List[str]]): New categorical columns to add.
            num (Optional[List[str]]): New numerical columns to add.
            bin (Optional[List[str]]): New binary columns to add.

        Raises:
            ValueError: If duplicate columns are detected after the update and
                `ignore_duplicate_cols` is False.
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

        col_ok, dup = self._check_column_overlap(self.categorical_columns,
                                                 self.numerical_columns,
                                                 self.binary_columns)
        if not col_ok:
            if not self.ignore_duplicate_cols:
                for c in dup:
                    logger.error(f'Find duplicate cols named `{c}`; set ignore_duplicate_cols=True to auto-resolve.')
                raise ValueError("Column overlap detected after update; aborting.")
            else:
                self._solve_duplicate_cols(dup)

    def _check_column_overlap(
        self,
        cat_cols: list[str] | None = None,
        num_cols: list[str] | None = None,
        bin_cols: list[str] | None = None,
    ) -> tuple[bool, list[str]]:
        """Check if the same column is categorized multiple times."""
        all_cols = []
        if cat_cols:
            all_cols += cat_cols
        if num_cols:
            all_cols += num_cols
        if bin_cols:
            all_cols += bin_cols

        if not all_cols:
            logger.warning("No columns specified; default to categorical.")
            return True, []

        counter = collections.Counter(all_cols)
        dup = [col for col, cnt in counter.items() if cnt > 1]
        return len(dup) == 0, dup

    def _solve_duplicate_cols(self, duplicate_cols: list[str]) -> None:
        """Duplicate columns are automatically renamed to distinguish them."""
        for col in duplicate_cols:
            logger.warning(f'Auto-resolving duplicate column `{col}`')
            if col in self.categorical_columns:
                self.categorical_columns.remove(col)
                self.categorical_columns.append(f'[cat]{col}')
            if col in self.numerical_columns:
                self.numerical_columns.remove(col)
                self.numerical_columns.append(f'[num]{col}')
            if col in self.binary_columns:
                self.binary_columns.remove(col)
                self.binary_columns.append(f'[bin]{col}')


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
        device: Union[str, torch.device] = "cpu",
        extractor: TransTabDataExtractor | None = None,
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
            ColType.NUMERICAL: TransTabNumEmbeddingEncoder(
                hidden_dim=out_dim
            ),
        }
        super().__init__(out_dim, metadata, col_pre_encoder_dict)
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.extractor = extractor if extractor is not None else TransTabDataExtractor(
            categorical_columns=None,
            numerical_columns=None,
            binary_columns=None,
        )
        self.align_layer = torch.nn.Linear(out_dim, out_dim, bias=False) if use_align_layer else torch.nn.Identity()
        self.to(self.device)

    def _encode_feat_dict(
        self,
        feat_dict: Dict[ColType, Tensor | Tuple[Tensor, ...]],
    ) -> Dict[ColType, Tensor]:
        feat_encoded: Dict[ColType, Tensor] = {}

        for col_type, feat in feat_dict.items():
            if col_type == ColType.NUMERICAL:
                col_ids, col_mask, raw_vals = feat  # [n_cols, L], [n_cols, L], [B, n_cols]
                token_emb = self.pre_encoder_dict[ColType.CATEGORICAL.value](col_ids)     # [n_cols, L, D]
                mask = col_mask.unsqueeze(-1)                                            # [n_cols, L, 1]
                token_emb = token_emb * mask
                col_emb = token_emb.sum(1) / mask.sum(1)                                 # [n_cols, D]
                num_emb = self.pre_encoder_dict[ColType.NUMERICAL.value](col_emb, raw_vals=raw_vals)
                feat_encoded[col_type] = num_emb                                         # [B, n_num, D]
            else:
                if isinstance(feat, tuple):
                    input_ids = feat[0]
                else:
                    input_ids = feat
                feat_encoded[col_type] = self.pre_encoder_dict[col_type.value](input_ids)

        return feat_encoded

    def _collect_masks_from_inputs(
        self,
        feat_dict: Dict[ColType, Tensor | Tuple[Tensor, ...]],
        emb_dict: Dict[ColType, Tensor],
        df_masks: Dict[str, Tensor] | None = None,
    ) -> Dict[ColType, Tensor]:
        masks: Dict[ColType, Tensor] = {}

        # Numerical: ones mask
        if ColType.NUMERICAL in emb_dict:
            B, n_num, _ = emb_dict[ColType.NUMERICAL].shape
            masks[ColType.NUMERICAL] = torch.ones(B, n_num, device=self.device)

        # From DataFrame path
        if df_masks is not None:
            if "cat_att_mask" in df_masks and ColType.CATEGORICAL in emb_dict:
                masks[ColType.CATEGORICAL] = df_masks["cat_att_mask"].to(self.device).float()
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
            raise ValueError("No features were encoded; check extractor/columns configuration.")

        all_emb = torch.cat(emb_list, dim=1)    # [B, total_seq_len, D]
        all_mask = torch.cat(mask_list, dim=1)  # [B, total_seq_len]
        return {"embedding": all_emb, "attention_mask": all_mask}

    def forward(
        self,
        x: Union[pd.DataFrame, Dict[ColType, Tensor | Tuple[Tensor, ...]]],
        *,
        shuffle: bool = False,
        align_and_concat: bool = True,
        return_dict: bool = False,
        requires_grad: bool = False,
    ) -> Union[Dict[str, Tensor], Dict[ColType, Tensor], Tensor]:
        grad_ctx = (lambda: torch.enable_grad()) if requires_grad else (lambda: torch.no_grad())
        with grad_ctx():
            if isinstance(x, pd.DataFrame):
                data = self.extractor(x, shuffle=shuffle)

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
                    return torch.cat(list(emb_dict.values()), dim=1) if len(emb_dict) > 0 else None

                df_masks = {
                    "cat_att_mask": data["cat_att_mask"] if data["cat_att_mask"] is not None else None,
                    "bin_att_mask": data["bin_att_mask"] if data["bin_att_mask"] is not None else None,
                }
                masks = self._collect_masks_from_inputs(feat_dict, emb_dict, df_masks=df_masks)
                return self._align_and_concat(emb_dict, masks)

            elif isinstance(x, dict):
                feat_dict = {k: v for k, v in x.items()}  # type: ignore
                emb_dict = self._encode_feat_dict(feat_dict)

                if not align_and_concat:
                    if return_dict:
                        return emb_dict
                    return torch.cat(list(emb_dict.values()), dim=1) if len(emb_dict) > 0 else None

                masks = self._collect_masks_from_inputs(feat_dict, emb_dict, df_masks=None)
                return self._align_and_concat(emb_dict, masks)

            else:
                raise TypeError("TransTabPreEncoder.forward: x must be a pandas.DataFrame or a feat_dict mapping.")

    def save(self, path: str) -> None:
        self.extractor.save(path)
        os.makedirs(path, exist_ok=True)
        encoder_path = os.path.join(path, INPUT_ENCODER_NAME)
        torch.save(self.state_dict(), encoder_path)
        logger.info(f"Saved pre_encoder (integrated) weights to {encoder_path}")

    def load(self, ckpt_dir: str) -> None:
        self.extractor.load(ckpt_dir)
        encoder_path = os.path.join(ckpt_dir, INPUT_ENCODER_NAME)
        try:
            state_dict = torch.load(encoder_path, map_location=self.device, weights_only=True)
        except TypeError:
            state_dict = torch.load(encoder_path, map_location=self.device)
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded pre_encoder (integrated) weights from {encoder_path}")
        logger.info(f" Missing keys: {missing}")
        logger.info(f" Unexpected keys: {unexpected}")
