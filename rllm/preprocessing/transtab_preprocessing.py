from __future__ import annotations
import os
from typing import Dict, Tuple
import collections
import json
import logging

import numpy as np
import torch
from torch import Tensor
from transformers import BertTokenizerFast

from rllm.types import ColType


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class TransTabDataExtractor:
    r"""TransTabDataExtractor: Transform TableData feat_dict into token IDs and value tensors,
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
            does not exist, "bert-base-uncased" is downloaded and saved here.
            Only used if `tokenizer` is not provided. (default: "./tokenizer")
        tokenizer (Optional[BertTokenizerFast]): A pre-initialized tokenizer instance.
            If provided, this takes precedence over `tokenizer_dir`. This is useful
            when you want to ensure consistency with TokenizerConfig used in TableData.
            (default: None)
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
        tokenizer: BertTokenizerFast | None = None,
        disable_tokenizer_parallel: bool = True,
        ignore_duplicate_cols: bool = False,
    ) -> None:
        # Initialize tokenizer
        # Priority: 1) Use provided tokenizer instance, 2) Load from tokenizer_dir, 3) Use default
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

        # Column grouping - preserve order while removing duplicates
        def deduplicate_preserve_order(seq):
            """Remove duplicates while preserving order."""
            seen = set()
            result = []
            for x in seq:
                if x not in seen:
                    seen.add(x)
                    result.append(x)
            return result

        self.categorical_columns = (deduplicate_preserve_order(categorical_columns) if categorical_columns else [])
        self.numerical_columns = (deduplicate_preserve_order(numerical_columns) if numerical_columns else [])
        self.binary_columns = (deduplicate_preserve_order(binary_columns) if binary_columns else [])
        self.ignore_duplicate_cols = ignore_duplicate_cols

        # Check and handle duplicate column names
        col_ok, dup = self._check_column_overlap(
            self.categorical_columns, self.numerical_columns, self.binary_columns
        )
        if not col_ok:
            if not self.ignore_duplicate_cols:
                for c in dup:
                    logger.error(f"Find duplicate cols named `{c}`; set ignore_duplicate_cols=True to auto-resolve.")
                raise ValueError("Column overlap detected; aborting.")
            else:
                self._solve_duplicate_cols(dup)

    def _process_from_feat_dict(
        self,
        feat_dict: Dict[ColType, Tensor | Tuple[Tensor, Tensor]],
        colname_token_ids: Dict[str, Tuple[Tensor, Tensor]] = None,
        shuffle: bool = False,
    ) -> dict[str, Tensor | None]:
        r"""Process pre-computed feat_dict from TableData into TransTab format.

        Args:
            feat_dict: Feature dictionary from TableData with:
                - ColType.TEXT: (input_ids [B, L], attention_mask [B, L]) - merged categorical columns
                - ColType.NUMERICAL: [B, n_num] - numerical values
                - ColType.BINARY: [B, n_bin] - binary values
            colname_token_ids: Dict mapping column names to (token_ids, attention_mask)
            shuffle: If True, shuffle column order within each type

        Returns:
            Dict with TransTab-compatible format
        """
        out: dict[str, Tensor | None] = {
            "x_num": None,
            "num_col_input_ids": None,
            "num_att_mask": None,
            "x_cat_input_ids": None,
            "cat_att_mask": None,
            "x_bin_input_ids": None,
            "bin_att_mask": None,
        }

        # Process TEXT columns (maps to TransTab's categorical)
        if ColType.TEXT in feat_dict:
            text_data = feat_dict[ColType.TEXT]
            if isinstance(text_data, tuple):
                out["x_cat_input_ids"] = text_data[0].long()  # [B, L]
                out["cat_att_mask"] = text_data[1].long()  # [B, L]
            else:
                raise ValueError("TEXT features must be tokenized (tuple of ids and mask)")

        # Process NUMERICAL columns
        if ColType.NUMERICAL in feat_dict:
            out["x_num"] = feat_dict[ColType.NUMERICAL].float()  # [B, n_num]

            # Extract numerical column names' token IDs from colname_token_ids
            # IMPORTANT: Use colname_token_ids order (not self.numerical_columns order)
            # to match the column order in feat_dict[ColType.NUMERICAL]
            if colname_token_ids is not None:
                # Get numerical columns in the order they appear in colname_token_ids
                num_cols = [c for c in colname_token_ids.keys() if c in self.numerical_columns]

                if shuffle:
                    np.random.shuffle(num_cols)
                if num_cols:
                    num_ids_list = [colname_token_ids[c][0] for c in num_cols]
                    num_mask_list = [colname_token_ids[c][1] for c in num_cols]
                    # Stack to [n_num, L]
                    out["num_col_input_ids"] = torch.stack(num_ids_list, dim=0).long()
                    out["num_att_mask"] = torch.stack(num_mask_list, dim=0).long()

        # Process BINARY columns
        if ColType.BINARY in feat_dict:
            x_bin = feat_dict[ColType.BINARY]  # [B, n_bin]

            # IMPORTANT: Use colname_token_ids order to match feat_dict column order
            if colname_token_ids is not None:
                bin_cols = [c for c in colname_token_ids.keys() if c in self.binary_columns]
                if shuffle:
                    np.random.shuffle(bin_cols)

                if bin_cols:
                    # For binary columns, only include column names where value is 1
                    # Need to generate per-row text based on which columns are active
                    batch_size = x_bin.shape[0]
                    bin_texts: list[str] = []

                    for i in range(batch_size):
                        # Get active columns for this row (value > 0.5, assuming 0/1 encoding)
                        active_cols = [
                            col for j, col in enumerate(bin_cols)
                            if j < x_bin.shape[1] and x_bin[i, j].item() > 0.5
                        ]
                        bin_texts.append(" ".join(active_cols))

                    # Tokenize the binary texts
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

    def __call__(
        self,
        shuffle: bool = False,
        feat_dict: Dict[ColType, Tensor | Tuple[Tensor, Tensor]] = None,
        colname_token_ids: Dict[str, Tuple[Tensor, Tensor]] = None,
    ) -> dict[str, Tensor | None]:
        # Use feat_dict-based processing
        if feat_dict is not None:
            return self._process_from_feat_dict(feat_dict, colname_token_ids, shuffle)
        # # Otherwise use original DataFrame-based processing
        # if df is None:
        #     raise ValueError("Either df or feat_dict must be provided")
        # cols = df.columns.tolist()
        # cat_cols = [
        #     c
        #     for c in cols
        #     if self.categorical_columns and c in self.categorical_columns
        # ]
        # num_cols = [
        #     c for c in cols if self.numerical_columns and c in self.numerical_columns
        # ]
        # bin_cols = [c for c in cols if self.binary_columns and c in self.binary_columns]

        # configured = bool(
        #     self.categorical_columns or self.numerical_columns or self.binary_columns
        # )
        # if not any((cat_cols, num_cols, bin_cols)):
        #     if configured:
        #         raise ValueError(
        #             "Configured cat/num/bin columns, but none matched DataFrame columns."
        #         )
        #     else:
        #         cat_cols = cols

        # if shuffle:
        #     np.random.shuffle(cat_cols)
        #     np.random.shuffle(num_cols)
        #     np.random.shuffle(bin_cols)

        # out: dict[str, Tensor | None] = {
        #     "x_num": None,
        #     "num_col_input_ids": None,
        #     "num_att_mask": None,
        #     "x_cat_input_ids": None,
        #     "cat_att_mask": None,
        #     "x_bin_input_ids": None,
        #     "bin_att_mask": None,
        # }

        # # Numerical columns
        # if num_cols:
        #     x_num_df = df[num_cols].infer_objects(copy=False).fillna(0)
        #     out["x_num"] = torch.tensor(x_num_df.values, dtype=torch.float32)
        #     tokens = self.tokenizer(
        #         num_cols,
        #         padding=True,
        #         truncation=True,
        #         add_special_tokens=False,
        #         return_tensors="pt",
        #     )
        #     out["num_col_input_ids"] = tokens["input_ids"]
        #     out["num_att_mask"] = tokens["attention_mask"]

        # # Categorical columns
        # if cat_cols:
        #     x_cat = df[cat_cols]
        #     mask = (~x_cat.isna()).astype(int)
        #     with pd.option_context("future.no_silent_downcasting", True):
        #         x_cat = x_cat.fillna("")
        #     x_cat = x_cat.astype(str)
        #     cat_texts: list[str] = []
        #     for values, flags in zip(x_cat.values, mask.values):
        #         tokens = [
        #             f"{col} {val}"
        #             for col, val, flag in zip(cat_cols, values, flags)
        #             if flag
        #         ]
        #         cat_texts.append(" ".join(tokens))
        #     tokens = self.tokenizer(
        #         cat_texts,
        #         padding=True,
        #         truncation=True,
        #         add_special_tokens=False,
        #         return_tensors="pt",
        #     )
        #     out["x_cat_input_ids"] = tokens["input_ids"]
        #     out["cat_att_mask"] = tokens["attention_mask"]

        # # Binary columns
        # if bin_cols:
        #     x_bin = df[bin_cols].copy()
        #     try:
        #         x_bin = x_bin.astype(str).map(lambda s: s.strip().lower())
        #     except AttributeError:
        #         x_bin = x_bin.astype(str).applymap(lambda s: s.strip().lower())

        #     POS = {"1", "true", "t", "yes", "y", "on"}
        #     NEG = {"0", "false", "f", "no", "n", "off", "", "nan", "none", "null"}
        #     pos_mask = x_bin.isin(POS)
        #     neg_mask = x_bin.isin(NEG)
        #     rem = x_bin.where(~(pos_mask | neg_mask))
        #     num = rem.apply(pd.to_numeric, errors="coerce")
        #     gt0 = num.fillna(0).astype(float) > 0
        #     x_bin_int = (pos_mask | gt0).astype(int)

        #     bin_texts: list[str] = []
        #     for values in x_bin_int.values:
        #         tokens = [col for col, flag in zip(bin_cols, values) if flag]
        #         bin_texts.append(" ".join(tokens))

        #     tokens = self.tokenizer(
        #         bin_texts,
        #         padding=True,
        #         truncation=True,
        #         add_special_tokens=False,
        #         return_tensors="pt",
        #     )
        #     if tokens["input_ids"].shape[1] > 0:
        #         out["x_bin_input_ids"] = tokens["input_ids"]
        #         out["bin_att_mask"] = tokens["attention_mask"]

        # return out
        else:
            raise ValueError("Must provide feat_dict for processing.")

    def save(self, path: str) -> None:
        r"""Save tokenizer and column grouping configuration to disk.

        The extractor state is saved under:
          {path}/{"extractor"}/tokenizer/  # tokenizer files
          {path}/{"extractor"}/{"extractor.json"}  # JSON of column lists

        Note: This always saves the current tokenizer instance, regardless of whether
        it was provided externally or loaded from tokenizer_dir during initialization.

        Args:
            path (str): Base directory in which to save extractor state.
        """
        save_path = os.path.join(path, "extractor")
        os.makedirs(save_path, exist_ok=True)

        tokenizer_path = os.path.join(save_path, "tokenizer")
        self.tokenizer.save_pretrained(tokenizer_path)

        coltype_path = os.path.join(save_path, "extractor.json")
        col_type_dict = {
            "categorical": self.categorical_columns,
            "numerical": self.numerical_columns,
            "binary": self.binary_columns,
        }
        with open(coltype_path, "w", encoding="utf-8") as f:
            json.dump(col_type_dict, f, ensure_ascii=False)

    def load(self, path: str) -> None:
        r"""Load tokenizer and column grouping from disk.

          Note: This replaces the current tokenizer instance with the one loaded from disk,
        even if a tokenizer was provided externally during initialization.

        Args:
            path (str): Base directory containing extractor state.
        """
        tokenizer_path = os.path.join(path, "extractor", "tokenizer")
        coltype_path = os.path.join(path, "extractor", "extractor.json")

        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
        with open(coltype_path, "r", encoding="utf-8") as f:
            col_type_dict = json.load(f)
        self.categorical_columns = col_type_dict.get("categorical", [])
        self.numerical_columns = col_type_dict.get("numerical", [])
        self.binary_columns = col_type_dict.get("binary", [])
        logger.info(f"Loaded extractor state from {coltype_path}")

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

        col_ok, dup = self._check_column_overlap(
            self.categorical_columns, self.numerical_columns, self.binary_columns
        )
        if not col_ok:
            if not self.ignore_duplicate_cols:
                for c in dup:
                    logger.error(
                        f"Find duplicate cols named `{c}`; set ignore_duplicate_cols=True to auto-resolve."
                    )
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
            logger.warning(f"Auto-resolving duplicate column `{col}`")
            if col in self.categorical_columns:
                self.categorical_columns.remove(col)
                self.categorical_columns.append(f"[cat]{col}")
            if col in self.numerical_columns:
                self.numerical_columns.remove(col)
                self.numerical_columns.append(f"[num]{col}")
            if col in self.binary_columns:
                self.binary_columns.remove(col)
                self.binary_columns.append(f"[bin]{col}")
