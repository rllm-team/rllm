from __future__ import annotations
from typing import Optional

import collections
import json
import os

import torch
from torch import Tensor
import numpy as np
import pandas as pd
from transformers import BertTokenizerFast
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

from rllm.types import ColType
from rllm.nn.pre_encoder import TransTabPreEncoder
from . import constants


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
        tokenizer_dir: str = "./transtab/tokenizer",
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

        if not any((cat_cols, num_cols, bin_cols)):
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
            x_num_df = df[num_cols].fillna(0).infer_objects(copy=False)
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
            x_bin = df[bin_cols].fillna(0).astype(int)
            bin_texts: list[str] = []
            for values in x_bin.values:
                tokens = [col for col, flag in zip(bin_cols, values) if flag]
                bin_texts.append(" ".join(tokens))
            tokens = self.tokenizer(
                bin_texts,
                padding=True,
                truncation=True,
                add_special_tokens=False,
                return_tensors="pt",
            )
            # Only include when there is at least one token per row
            if tokens["input_ids"].shape[1] > 0:
                out["x_bin_input_ids"] = tokens["input_ids"]
                out["bin_att_mask"] = tokens["attention_mask"]

        return out

    def save(self, path: str) -> None:
        r"""Save tokenizer and column grouping configuration to disk.

        The extractor state is saved under:
          {path}/{constants.EXTRACTOR_STATE_DIR}/tokenizer/  # tokenizer files
          {path}/{constants.EXTRACTOR_STATE_DIR}/{constants.EXTRACTOR_STATE_NAME}  # JSON of column lists

        Args:
            path (str): Base directory in which to save extractor state.
        """
        save_path = os.path.join(path, constants.EXTRACTOR_STATE_DIR)
        os.makedirs(save_path, exist_ok=True)

        tokenizer_path = os.path.join(save_path, constants.TOKENIZER_DIR)
        self.tokenizer.save_pretrained(tokenizer_path)

        coltype_path = os.path.join(save_path, constants.EXTRACTOR_STATE_NAME)
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
        tokenizer_path = os.path.join(path, constants.EXTRACTOR_STATE_DIR, constants.TOKENIZER_DIR)
        coltype_path = os.path.join(path, constants.EXTRACTOR_STATE_DIR, constants.EXTRACTOR_STATE_NAME)

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


class TransTabDataProcessor(torch.nn.Module):
    r"""TransTabDataProcessor: Combine TransTabDataExtractor with TransTabPreEncoder,
    then apply feature alignment and concatenation as described in
    `"TransTab: Learning Transferable Tabular Transformers Across Tables"`
    <https://arxiv.org/abs/2205.09328>`_ paper.

    This module extracts raw tensors via TransTabDataExtractor, encodes them
    through the provided TransTabPreEncoder, applies a linear alignment layer,
    and concatenates all feature embeddings and masks into a single output.

    Args:
        pre_encoder (TransTabPreEncoder): Pre-encoder module for token embedding.
        out_dim (int): Output embedding dimension used by the alignment layer.
        device (Union[str, torch.device]): Device for model parameters and data.
            (default: "cpu")
    """

    def __init__(
        self,
        pre_encoder: TransTabPreEncoder,
        out_dim: int,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.pre_encoder = pre_encoder.to(device)
        self.extractor = TransTabDataExtractor(
            categorical_columns=None,
            numerical_columns=None,
            binary_columns=None,
        )
        # align_layer mirrors original FeatureProcessor
        self.align_layer = torch.nn.Linear(out_dim, out_dim, bias=False).to(device)
        self.device = device

    def forward(
        self,
        df: pd.DataFrame,
        shuffle: bool = False,
    ) -> dict[str, Tensor]:
        r"""Process a DataFrame into unified embeddings and attention mask.

        Steps:
          1. Extract raw tensors and masks via TransTabDataExtractor.
          2. Encode features with TransTabPreEncoder.
          3. Align each feature embedding type via a linear layer.
          4. Concatenate embeddings and corresponding masks.

        Args:
            df (pd.DataFrame): Input data frame.
            shuffle (bool): If True, randomly shuffle column order in extraction.
                (default: False)

        Returns:
            Dict[str, Tensor] with keys:
              - "embedding": Float tensor of shape (batch_size, total_seq_len, out_dim)
              - "attention_mask": Float tensor of shape (batch_size, total_seq_len)
        """
        data = self.extractor(df, shuffle=shuffle)

        feat_dict: dict[ColType, Tensor] = {}
        if data["x_cat_input_ids"] is not None:
            feat_dict[ColType.CATEGORICAL] = data["x_cat_input_ids"].to(self.device)
        if data["x_bin_input_ids"] is not None:
            feat_dict[ColType.BINARY] = data["x_bin_input_ids"].to(self.device)
        if data["x_num"] is not None:
            feat_dict[ColType.NUMERICAL] = (
                data["num_col_input_ids"].to(self.device),
                data["num_att_mask"].to(self.device),
                data["x_num"].to(self.device),
            )

        emb_dict = self.pre_encoder(feat_dict, return_dict=True)

        num_emb = emb_dict.get(ColType.NUMERICAL)
        if num_emb is not None:
            num_emb = self.align_layer(num_emb)
        cat_emb = emb_dict.get(ColType.CATEGORICAL)
        if cat_emb is not None:
            cat_emb = self.align_layer(cat_emb)
        bin_emb = emb_dict.get(ColType.BINARY)
        if bin_emb is not None:
            bin_emb = self.align_layer(bin_emb)

        emb_list: list[Tensor] = []
        mask_list: list[Tensor] = []
        if num_emb is not None:
            emb_list.append(num_emb)
            mask_list.append(
                torch.ones(num_emb.shape[0], num_emb.shape[1], device=self.device)
            )
        if cat_emb is not None:
            emb_list.append(cat_emb)
            mask_list.append(data["cat_att_mask"].to(self.device).float())
        if bin_emb is not None:
            emb_list.append(bin_emb)
            mask_list.append(data["bin_att_mask"].to(self.device).float())

        all_emb = torch.cat(emb_list, dim=1)
        all_mask = torch.cat(mask_list, dim=1)
        return {"embedding": all_emb, "attention_mask": all_mask}

    def save(self, path: str) -> None:
        r"""Save extractor configuration and pre_encoder weights to disk.

        The following files/directories are created under `path`:
          - extractor/: saved by TransTabDataExtractor.save()
          - input_encoder.bin: serialized state_dict of `pre_encoder`

        Args:
            path (str): Base directory in which to save processor state.
        """
        # 1) Saving extractor state
        self.extractor.save(path)

        # 2) Save the weights of pre_encoder
        os.makedirs(path, exist_ok=True)
        encoder_path = os.path.join(path, constants.INPUT_ENCODER_NAME)
        torch.save(self.pre_encoder.state_dict(), encoder_path)
        logger.info(f"Saved pre_encoder weights to {encoder_path}")

    def load(self, ckpt_dir: str) -> None:
        r"""Load extractor configuration and pre_encoder weights from disk.

        Expects the structure created by `save()`:
          - extractor/ (tokenizer + column JSON)
          - input_encoder.bin

        Args:
            ckpt_dir (str): Directory containing saved state.
        """
        # 1) Restore extractor state
        self.extractor.load(ckpt_dir)

        # 2) Restore the weights of pre_encoder
        encoder_path = os.path.join(ckpt_dir, constants.INPUT_ENCODER_NAME)
        state_dict = torch.load(encoder_path, map_location=self.device, weights_only=True)
        missing, unexpected = self.pre_encoder.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded pre_encoder weights from {encoder_path}")
        logger.info(f" Missing keys: {missing}")
        logger.info(f" Unexpected keys: {unexpected}")


def _get_activation_fn(activation):
    if activation == "relu":
        return torch.nn.functional.relu
    elif activation == "gelu":
        return torch.nn.functional.gelu
    elif activation == 'selu':
        return torch.nn.functional.selu
    elif activation == 'leakyrelu':
        return torch.nn.functional.leaky_relu
    raise RuntimeError("activation should be relu/gelu/selu/leakyrelu, not {}".format(activation))


class TransTabTransformerEncoderLayer(torch.nn.Module):
    r"""The TransTabTransformerEncoderLayer module introduced in
    `"TransTab: Learning Transferable Tabular Transformers Across Tables"`
    <https://arxiv.org/abs/2205.09328>`_ paper.

    This layer implements a single Transformer encoder block customized for
    tabular transfer learning. It combines multi-head self-attention, a gated
    feedforward network, optional pre-/post-layer normalization, residual
    connections, and dropout to capture complex feature interactions in table
    data.

    Args:
        d_model (int): Dimensionality of input and output feature vectors. (default: required)
        nhead (int): Number of attention heads. (default: required)
        dim_feedforward (int): Hidden dimensionality of the feedforward network.
            If None, defaults to `d_model`. (default: 2048)
        dropout (float): Dropout probability applied in attention and
            feedforward sublayers. (default: 0.1)
        activation (Union[str, Callable]): Activation function for the
            feedforward network, specified as a callable or a string name
            (e.g., "relu"). (default: torch.nn.functional.relu)
        layer_norm_eps (float): Epsilon value for all LayerNorm layers to
            ensure numerical stability. (default: 1e-5)
        batch_first (bool): If True, input and output tensors are expected
            in shape `(batch_size, seq_len, d_model)`; otherwise
            `(seq_len, batch_size, d_model)`. (default: True)
        norm_first (bool): If True, apply LayerNorm before self-attention
            and feedforward; otherwise apply after the residual connection.
            (default: False)
        use_layer_norm (bool): Whether to include LayerNorm layers in each
            sub-block. (default: True)
        device (Optional[torch.device]): Device on which to allocate layer
            parameters. (default: None)
        dtype (Optional[torch.dtype]): Data type for layer parameters.
            (default: None)
    """

    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=torch.nn.functional.relu,
                 layer_norm_eps=1e-5, batch_first=True, norm_first=False,
                 device=None, dtype=None, use_layer_norm=True) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = torch.nn.MultiheadAttention(d_model, nhead, batch_first=batch_first, **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        # Implementation of gates
        self.gate_linear = torch.nn.Linear(d_model, 1, bias=False)
        self.gate_act = torch.nn.Sigmoid()

        self.norm_first = norm_first
        self.use_layer_norm = use_layer_norm

        if self.use_layer_norm:
            self.norm1 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        key_padding_mask = ~key_padding_mask.bool()
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        g = self.gate_act(self.gate_linear(x))
        h = self.linear1(x)
        h = h * g   # add gate
        h = self.linear2(self.dropout(self.activation(h)))
        return self.dropout2(h)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = torch.nn.functional.relu
        super().__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=None, **kwargs) -> Tensor:
        r"""Pass the input through this encoder layer.

        Args:
            src (Tensor): Input tensor of shape
                `(batch_size, seq_len, d_model)` if `batch_first=True`,
                else `(seq_len, batch_size, d_model)`.
            src_mask (Optional[Tensor]): Attention mask of shape
                `(seq_len, seq_len)` or broadcastable. (default: None)
            src_key_padding_mask (Optional[Tensor]): Padding mask of shape
                `(batch_size, seq_len)` where True values are ignored. (default: None)
            is_causal (Optional[bool]): Unused; present for API compatibility.

        Returns:
            Tensor: Output tensor of the same shape as `src`, after applying
            self-attention, gated feedforward, residual connections, and
            optional layer normalization.
        """

        x = src
        if self.use_layer_norm:
            if self.norm_first:
                x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
                x = x + self._ff_block(self.norm2(x))
            else:
                x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
                x = self.norm2(x + self._ff_block(x))

        else:  # do not use layer norm
            x = x + self._sa_block(x, src_mask, src_key_padding_mask)
            x = x + self._ff_block(x)
        return x


class TransTabConv(torch.nn.Module):
    r"""The TransTabConv introduced in the
    `"TransTab: Learning Transferable Tabular Transformers Across Tables"`
    <https://arxiv.org/abs/2205.09328>`_ paper.

    This module stacks one or more Transformer encoder layers to process a
    sequence of column embeddings, enabling transfer of learned patterns
    across different tables.

    Args:
        hidden_dim (int): Dimensionality of input/output embeddings (d_model). (default: 128)
        num_layer (int): Total number of Transformer encoder layers to apply.
            If >1, the first layer is a custom TransTabTransformerEncoderLayer
            and the rest are standard clones. (default: 2)
        num_attention_head (int): Number of attention heads per layer. (default: 2)
        hidden_dropout_prob (float): Dropout probability in attention and
            feedforward sublayers. (default: 0.0)
        ffn_dim (int): Inner feedforward dimension (typically ≥ hidden_dim). (default: 256)
        activation (str): Activation function for the feedforward network,
            e.g. "relu". (default: "relu")
        layer_norm_eps (float): Epsilon for LayerNorm layers. (default: 1e-5)
        norm_first (bool): If True, apply LayerNorm before sublayers;
            otherwise after residuals. (default: False)
        use_layer_norm (bool): Whether to include LayerNorm in each layer.
            (default: True)
        batch_first (bool): If True, expect inputs as (batch, seq, dim);
            else (seq, batch, dim). (default: True)
        device (torch.device, optional): Device for model parameters.
        dtype (torch.dtype, optional): Dtype for model parameters.

    Returns:
        torch.Tensor: Output of shape (batch_size, seq_len, hidden_dim).
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layer: int = 2,
        num_attention_head: int = 2,
        hidden_dropout_prob: float = 0.0,
        ffn_dim: int = 256,
        activation: str = 'relu',
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
        use_layer_norm: bool = True,
        batch_first: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        # First layer: one custom TransTabTransformerEncoderLayer
        first_layer = TransTabTransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_attention_head,
            dropout=hidden_dropout_prob,
            dim_feedforward=ffn_dim,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            norm_first=norm_first,
            use_layer_norm=use_layer_norm,
            batch_first=batch_first,
            **factory_kwargs,
        )

        self.transformer_layers = torch.nn.ModuleList([first_layer])

        # If more than one layer, stack the rest in a TransformerEncoder
        if num_layer > 1:
            encoder_layer = TransTabTransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_attention_head,
                dropout=hidden_dropout_prob,
                dim_feedforward=ffn_dim,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
                norm_first=norm_first,
                use_layer_norm=use_layer_norm,
                batch_first=batch_first,
                **factory_kwargs,
            )
            stacked = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layer - 1)
            self.transformer_layers.append(stacked)

    def forward(
        self,
        embedding: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            embedding: Tensor of shape (batch_size, seq_len, hidden_dim)
            attention_mask: Bool or float mask of shape (batch_size, seq_len),
                            where True/1 indicates keep and False/0 indicates mask.
        Returns:
            Tensor of shape (batch_size, seq_len, hidden_dim)
        """
        x = embedding
        # iterate through each transformer module
        for layer in self.transformer_layers:
            # both TransTabTransformerEncoderLayer and torch.nn.TransformerEncoder accept
            # src_key_padding_mask argument
            x = layer(x, src_key_padding_mask=attention_mask)
        return x
