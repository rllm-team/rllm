from __future__ import annotations
from typing import Optional, Dict, List, Any, Union, Callable

import os
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import ModuleList, Parameter

from transformers import BertTokenizerFast
from rllm.types import ColType
from rllm.nn.pre_encoder import TransTabPreEncoder  
from rllm.data.table_data import TableData
import os, pdb
import math
import collections
import json

from loguru import logger
from transformers import BertTokenizer, BertTokenizerFast

import torch.nn.init as nn_init
import torch.nn.functional as F
import numpy as np
import pandas as pd
from . import constants


class TransTabDataExtractor:
    """
    Extract raw DataFrame columns into token IDs and value tensors,
    matching original TransTabFeatureExtractor behavior.

    Converts the original columns of the input pandas.DataFrame divided by column type (numeric/categorical/binary) 
    into PyTorch tensors that can be directly consumed by subsequent models
    """
    def __init__(
        self,
        categorical_columns: list[str] | None = None,
        numerical_columns: list[str]   | None = None,
        binary_columns:    list[str]   | None = None,
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
        self.numerical_columns   = list(set(numerical_columns))   if numerical_columns   else []
        self.binary_columns      = list(set(binary_columns))      if binary_columns      else []
        self.ignore_duplicate_cols = ignore_duplicate_cols

        # 检查并处理重名列
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
        cols = df.columns.tolist()
        # Select columns by configured lists
        cat_cols = [c for c in cols if self.categorical_columns and c in self.categorical_columns]
        num_cols = [c for c in cols if self.numerical_columns   and c in self.numerical_columns]
        bin_cols = [c for c in cols if self.binary_columns      and c in self.binary_columns]

        # Default: treat all as categorical if none specified
        if not any((cat_cols, num_cols, bin_cols)):
            cat_cols = cols

        # Shuffle column order as original
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
            x_num_df = df[num_cols].fillna(0).infer_objects(copy=False) #Futurewarning
            out["x_num"] = torch.tensor(x_num_df.values, dtype=torch.float32)
            tokens = self.tokenizer(
                num_cols,
                padding=True,
                truncation=True,
                add_special_tokens=False,
                return_tensors="pt",
            )
            out["num_col_input_ids"] = tokens["input_ids"]
            out["num_att_mask"]      = tokens["attention_mask"]

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
            out["cat_att_mask"]    = tokens["attention_mask"]

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
                out["bin_att_mask"]    = tokens["attention_mask"]

        return out

    def save(self, path: str) -> None:
        """Save tokenizer & column grouping to disk."""
        save_path = os.path.join(path, constants.EXTRACTOR_STATE_DIR)
        os.makedirs(save_path, exist_ok=True)

        # save tokenizer
        tokenizer_path = os.path.join(save_path, constants.TOKENIZER_DIR)
        self.tokenizer.save_pretrained(tokenizer_path)

        # save column lists
        coltype_path = os.path.join(save_path, constants.EXTRACTOR_STATE_NAME)
        col_type_dict = {
            'categorical': self.categorical_columns,
            'numerical':   self.numerical_columns,
            'binary':      self.binary_columns,
        }
        with open(coltype_path, 'w', encoding='utf-8') as f:
            json.dump(col_type_dict, f, ensure_ascii=False)

    def load(self, path: str) -> None:
        """Load tokenizer & column grouping from disk."""
        tokenizer_path = os.path.join(path, constants.EXTRACTOR_STATE_DIR, constants.TOKENIZER_DIR)
        coltype_path   = os.path.join(path, constants.EXTRACTOR_STATE_DIR, constants.EXTRACTOR_STATE_NAME)

        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
        with open(coltype_path, 'r', encoding='utf-8') as f:
            col_type_dict = json.load(f)
        self.categorical_columns = col_type_dict.get('categorical', [])
        self.numerical_columns   = col_type_dict.get('numerical', [])
        self.binary_columns      = col_type_dict.get('binary', [])
        logger.info(f'Loaded extractor state from {coltype_path}')

    def update(
        self,
        cat: list[str] | None = None,
        num: list[str] | None = None,
        bin: list[str] | None = None,
    ) -> None:
        """Dynamically extend column lists,并重新检查重名."""
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
        """检测同一列是否被多次归类。"""
        all_cols = []
        if cat_cols: all_cols += cat_cols
        if num_cols: all_cols += num_cols
        if bin_cols: all_cols += bin_cols

        if not all_cols:
            logger.warning("No columns specified; default to categorical.")
            return True, []

        counter = collections.Counter(all_cols)
        dup = [col for col, cnt in counter.items() if cnt > 1]
        return len(dup) == 0, dup

    def _solve_duplicate_cols(self, duplicate_cols: list[str]) -> None:
        """对重名列自动重命名以示区分。"""
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

class TransTabDataProcessor(nn.Module):
    """
    Combine TransTabDataExtractor with TransTabPreEncoder,
    then apply avg-mask, alignment and concatenate embeddings/masks.

    The original tensors extracted by TransTabDataExtractor from the upstream DataFrame are gradually sent to TransTabPreEncoder, 
    and then various feature embeddings are "linearly aligned" and concatenated to finally obtain a unified feature embedding and attention mask.
    """
    def __init__(
        self,
        pre_encoder: TransTabPreEncoder,
        out_dim: int,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.pre_encoder = pre_encoder.to(device)
        self.extractor    = TransTabDataExtractor(
            categorical_columns=None,
            numerical_columns=None,
            binary_columns=None,
        )
        # align_layer mirrors original FeatureProcessor
        self.align_layer  = nn.Linear(out_dim, out_dim, bias=False).to(device)
        self.device       = device

    def forward(
        self,
        df: pd.DataFrame,
        shuffle: bool = False,
    ) -> dict[str, Tensor]:
        # 1) Extract raw tensors and masks
        data = self.extractor(df, shuffle=shuffle)

        # 2) Prepare feature dict for PreEncoder
        feat_dict: dict[ColType, Tensor] = {}
        # categorical and binary features go through Token embedding
        if data["x_cat_input_ids"] is not None:
            feat_dict[ColType.CATEGORICAL] = data["x_cat_input_ids"].to(self.device)
        if data["x_bin_input_ids"] is not None:
            feat_dict[ColType.BINARY] = data["x_bin_input_ids"].to(self.device)
        # numerical features: raw values only
        if data["x_num"] is not None:
            feat_dict[ColType.NUMERICAL] = (
                data["num_col_input_ids"].to(self.device),
                data["num_att_mask"].to(self.device),
                data["x_num"].to(self.device),
            )

        # 3) Run through PreEncoder to get embeddings dict
        emb_dict = self.pre_encoder(feat_dict, return_dict=True)

        # 4) Align each feature embedding type
        num_emb = emb_dict.get(ColType.NUMERICAL)
        if num_emb is not None:
            num_emb = self.align_layer(num_emb)
        cat_emb = emb_dict.get(ColType.CATEGORICAL)
        if cat_emb is not None:
            cat_emb = self.align_layer(cat_emb)
        bin_emb = emb_dict.get(ColType.BINARY)
        if bin_emb is not None:
            bin_emb = self.align_layer(bin_emb)

        # 5) Concatenate embeddings and build attention mask
        emb_list: list[Tensor] = []
        mask_list: list[Tensor] = []
        # numerical: full attention
        if num_emb is not None:
            emb_list.append(num_emb)
            mask_list.append(
                torch.ones(num_emb.shape[0], num_emb.shape[1], device=self.device)
            )
        # categorical: use token masks
        if cat_emb is not None:
            emb_list.append(cat_emb)
            mask_list.append(data["cat_att_mask"].to(self.device).float())
        # binary: use token masks
        if bin_emb is not None:
            emb_list.append(bin_emb)
            mask_list.append(data["bin_att_mask"].to(self.device).float())

        all_emb  = torch.cat(emb_list, dim=1)
        all_mask = torch.cat(mask_list, dim=1)
        return {"embedding": all_emb, "attention_mask": all_mask}
    
    def save(self, path: str) -> None:
        """
        保存 extractor 的列配置和 tokenizer，以及 pre_encoder 的权重。
        最终会在 path 下生成：
          extractor/           (由 extractor.save 生成，包含 tokenizer/ 和 extractor.json)
          input_encoder.bin    （保存 pre_encoder.state_dict()）
        """
        # 1) 保存 extractor 状态
        self.extractor.save(path)

        # 2) 保存 pre_encoder 的权重
        os.makedirs(path, exist_ok=True)
        encoder_path = os.path.join(path, constants.INPUT_ENCODER_NAME)
        torch.save(self.pre_encoder.state_dict(), encoder_path)
        logger.info(f"Saved pre_encoder weights to {encoder_path}")

    def load(self, ckpt_dir: str) -> None:
        """
        加载 extractor 的列配置和 tokenizer，以及 pre_encoder 的权重。
        假定目录结构同 save() 所写。
        """
        # 1) 恢复 extractor 状态
        self.extractor.load(ckpt_dir)

        # 2) 恢复 pre_encoder 的权重
        encoder_path = os.path.join(ckpt_dir, constants.INPUT_ENCODER_NAME)
        state_dict = torch.load(encoder_path, map_location=self.device, weights_only=True)
        missing, unexpected = self.pre_encoder.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded pre_encoder weights from {encoder_path}")
        logger.info(f" Missing keys: {missing}")
        logger.info(f" Unexpected keys: {unexpected}")

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == 'selu':
        return F.selu
    elif activation == 'leakyrelu':
        return F.leaky_relu
    raise RuntimeError("activation should be relu/gelu/selu/leakyrelu, not {}".format(activation))

class TransTabTransformerLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=True, norm_first=False,
                 device=None, dtype=None, use_layer_norm=True) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        # Implementation of gates
        self.gate_linear = nn.Linear(d_model, 1, bias=False)
        self.gate_act = nn.Sigmoid()

        self.norm_first = norm_first
        self.use_layer_norm = use_layer_norm

        if self.use_layer_norm:
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        src = x
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
        h = h * g # add gate
        h = self.linear2(self.dropout(self.activation(h)))
        return self.dropout2(h)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, src, src_mask= None, src_key_padding_mask= None, is_causal=None, **kwargs) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        x = src
        if self.use_layer_norm:
            if self.norm_first:
                x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
                x = x + self._ff_block(self.norm2(x))
            else:
                x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
                x = self.norm2(x + self._ff_block(x))

        else: # do not use layer norm
                x = x + self._sa_block(x, src_mask, src_key_padding_mask)
                x = x + self._ff_block(x)
        return x

class TransTabCLSToken(nn.Module):
    '''add a learnable cls token embedding at the end of each sequence.
    '''
    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.weight = nn.Parameter(Tensor(hidden_dim))
        nn_init.uniform_(self.weight, a=-1/math.sqrt(hidden_dim),b=1/math.sqrt(hidden_dim))
        self.hidden_dim = hidden_dim

    def expand(self, *leading_dimensions):
        new_dims = (1,) * (len(leading_dimensions)-1)
        return self.weight.view(*new_dims, -1).expand(*leading_dimensions, -1)

    def forward(self, embedding, attention_mask=None, **kwargs) -> Tensor:
        embedding = torch.cat([self.expand(len(embedding), 1), embedding], dim=1)
        outputs = {'embedding': embedding}
        if attention_mask is not None:
            attention_mask = torch.cat([torch.ones(attention_mask.shape[0],1).to(attention_mask.device), attention_mask], 1)
        outputs['attention_mask'] = attention_mask
        return outputs


class TransTabConv(nn.Module):
    """
    Multi‐layer Transformer encoder for tabular inputs.
    Mirrors original TransTabEncoder behavior.
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

        # First layer: one custom TransTabTransformerLayer
        first_layer = TransTabTransformerLayer(
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

        self.transformer_layers = nn.ModuleList([first_layer])

        # If more than one layer, stack the rest in a TransformerEncoder
        if num_layer > 1:
            encoder_layer = TransTabTransformerLayer(
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
            stacked = nn.TransformerEncoder(encoder_layer, num_layers=num_layer - 1)
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
            # both TransTabTransformerLayer and nn.TransformerEncoder accept
            # src_key_padding_mask argument
            x = layer(x, src_key_padding_mask=attention_mask)
        return x
    






class TransTabModel(nn.Module):
    """
    A full TransTab model for downstream tasks.
    Combines:
      1) Table → token ids/values (TransTabDataExtractor)
      2) token ids/values → embeddings + masks (TransTabDataProcessor)
      3) prepend [CLS], run through Transformer stack (TransTabConv + TransTabCLSToken)
      4) extract final CLS embedding

    API compatible with original TransTabModel.
    """
    def __init__(
        self,
        categorical_columns: List[str]   = None,
        numerical_columns:   List[str]   = None,
        binary_columns:      List[str]   = None,
        hidden_dim:          int         = 128,
        num_layer:           int         = 2,
        num_attention_head:  int         = 8,
        hidden_dropout_prob: float       = 0.1,
        ffn_dim:             int         = 256,
        activation:          str         = 'relu',
        device:              Union[str, torch.device] = 'cuda:0',
        **kwargs,  # 允许向 DataExtractor 透传额外配置
    ) -> None:
        super().__init__()

        # 1) 记录并去重各类列名
        self.categorical_columns = list(set(categorical_columns)) if categorical_columns else None
        self.numerical_columns   = list(set(numerical_columns))   if numerical_columns   else None
        self.binary_columns      = list(set(binary_columns))      if binary_columns      else None

        # 2) 初始化 DataExtractor（保留所有 **kwargs 配置）
        self.extractor = TransTabDataExtractor(
            categorical_columns=self.categorical_columns,
            numerical_columns=self.numerical_columns,
            binary_columns=self.binary_columns,
            **kwargs,
        )

        # 3) 初始化 PreEncoder（用于 DataProcessor），metadata 提供空列表映射
        metadata = {
            ColType.CATEGORICAL: [],
            ColType.BINARY:      [],
            ColType.NUMERICAL:   [],
        }
        self.pre_encoder = TransTabPreEncoder(
            out_dim=hidden_dim,
            metadata=metadata,
            vocab_size=self.extractor.tokenizer.vocab_size,
            padding_idx=self.extractor.tokenizer.pad_token_id,
            hidden_dropout_prob=hidden_dropout_prob,
            layer_norm_eps=1e-5,
        )

        # 4) 初始化 DataProcessor，将 extractor 与 pre_encoder 组合
        self.data_processor = TransTabDataProcessor(
            pre_encoder=self.pre_encoder,
            out_dim=hidden_dim,
            device=device,
        )

        # 5) 构建多层 Transformer 编码器（使用 TransTabConv）
        self.encoder = TransTabConv(
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            num_attention_head=num_attention_head,
            hidden_dropout_prob=hidden_dropout_prob,
            ffn_dim=ffn_dim,
            activation=activation,
            layer_norm_eps=1e-5,
            norm_first=False,
            use_layer_norm=True,
            batch_first=True,
            device=device,
        )

        # 6) CLS token 模块，用于在序列最前端插入可学习向量
        self.cls_token = TransTabCLSToken(hidden_dim=hidden_dim)

        self.device = device
        self.to(device)

    def forward(
        self,
        x: Union[pd.DataFrame, TableData],
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: 一批样本（pd.DataFrame 或 TableData）
        y: 可选标签（仅作占位，基类忽略）
        返回: 最终的 [CLS] 向量，shape = (batch, hidden_dim)
        """
        # —— 0) 支持 TableData 类型输入
        if isinstance(x, TableData):
            df = x.df
        else:
            df = x

        # —— 1) DataProcessor 得到 embedding + mask
        proc_out = self.data_processor(df)
        emb  = proc_out['embedding']       # (batch, seq_len, hidden_dim)
        mask = proc_out['attention_mask']  # (batch, seq_len)

        # —— 2) 在最前面加上 CLS token
        cls_out = self.cls_token(emb, attention_mask=mask)
        emb2  = cls_out['embedding']       # (batch, seq_len+1, hidden_dim)
        mask2 = cls_out['attention_mask']  # (batch, seq_len+1)

        # —— 3) Transformer 编码
        enc_out = self.encoder(embedding=emb2, attention_mask=mask2)

        # —— 4) 取出第一个位置（CLS）作为样本表示
        final_cls = enc_out[:, 0, :]       # (batch, hidden_dim)
        return final_cls

    def save(self, ckpt_dir: str) -> None:
        """
        保存整个模型：
          1) data_processor.save → extractor/tokenizer + pre_encoder
          2) 保存本模型（encoder + cls_token）的 state_dict
        """
        # 1) 保存 extractor + pre_encoder
        self.data_processor.save(ckpt_dir)

        # 2) 保存模型权重
        os.makedirs(ckpt_dir, exist_ok=True)
        model_path = os.path.join(ckpt_dir, constants.WEIGHTS_NAME)
        torch.save(self.state_dict(), model_path)
        logger.info(f"Saved TransTabModel weights to {model_path}")

    def load(self, ckpt_dir: str) -> None:
        """
        加载整个模型：
        1) 恢复 extractor/tokenizer + pre_encoder
        2) 同步列映射到模型属性
        3) 加载 encoder + cls_token 的 state_dict（先到 CPU）
        4) 把模型搬到 self.device
        """
        # 1) 恢复 extractor + pre_encoder
        self.data_processor.load(ckpt_dir)

        # 2) 同步列映射
        self.categorical_columns = self.data_processor.extractor.categorical_columns
        self.numerical_columns   = self.data_processor.extractor.numerical_columns
        self.binary_columns      = self.data_processor.extractor.binary_columns

        # 3) 加载模型权重到 CPU
        model_path = os.path.join(ckpt_dir, constants.WEIGHTS_NAME)
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded TransTabModel weights from {model_path}")
        logger.info(f" Missing keys: {missing}")
        logger.info(f" Unexpected keys: {unexpected}")

        # 4) 搬到目标设备
        self.to(self.device)

    def update(self, config: Dict[str, Any]) -> None:
        """
        动态更新列映射，和（可选）分类头类别数：
          - 'cat' / 'num' / 'bin' : 新的列列表
          - 'num_class'           : 新的分类数（只有在子类定义 clf 时生效）
        """
        # 更新列映射
        col_map = {k: v for k, v in config.items() if k in ('cat', 'num', 'bin')}
        if col_map:
            self.data_processor.extractor.update(**col_map)
            self.categorical_columns = self.data_processor.extractor.categorical_columns
            self.numerical_columns   = self.data_processor.extractor.numerical_columns
            self.binary_columns      = self.data_processor.extractor.binary_columns
            logger.info("Updated column mappings in TransTabModel.")

        # 可选：更新分类头
        if 'num_class' in config:
            self._adapt_to_new_num_class(config['num_class'])

    def _adapt_to_new_num_class(self, num_class: int) -> None:
        """
        如果本模型（或子类）定义了 self.clf，则在类别数变化时重建分类头和 loss_fn。
        """
        if not hasattr(self, 'clf') or num_class == getattr(self, 'num_class', None):
            return
        self.num_class = num_class
        # 重建分类头
        self.clf = TransTabLinearClassifier(num_class=num_class,
                                            hidden_dim=self.cls_token.hidden_dim)
        self.clf.to(self.device)
        # 重建 loss
        if num_class > 2:
            self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        logger.info(f"Rebuilt classifier for num_class={num_class}.")


    '''
    def forward(
        self,
        x: Union[pd.DataFrame, 'TableData'],
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: 一批样本（pd.DataFrame 或 TableData）
        y: 可选标签（仅作占位，基类忽略）
        返回: 最终的 [CLS] 向量，shape = (batch, hidden_dim)
        """
        # —— 1) DataProcessor 得到 embedding + mask
        proc_out = self.data_processor(x)
        emb  = proc_out['embedding']       # (batch, seq_len, hidden_dim)
        mask = proc_out['attention_mask']  # (batch, seq_len)

        # —— 2) 在最前面加上 CLS token
        cls_out = self.cls_token(emb, attention_mask=mask)
        emb2  = cls_out['embedding']       # (batch, seq_len+1, hidden_dim)
        mask2 = cls_out['attention_mask']  # (batch, seq_len+1)

        # —— 3) Transformer 编码
        enc_out = self.encoder(embedding=emb2, attention_mask=mask2)

        # —— 4) 取出第一个位置（CLS）作为样本表示
        final_cls = enc_out[:, 0, :]       # (batch, hidden_dim)
        return final_cls
    '''

    '''
    def save(self, ckpt_dir: str):
        os.makedirs(ckpt_dir, exist_ok=True)
        # 1) 保存模型主干及分类器权重
        torch.save(self.state_dict(), os.path.join(ckpt_dir, WEIGHTS_NAME))
        # 2) 保存特征抽取器配置
        self.data_processor.extractor.save(ckpt_dir)

    def load(self, ckpt_dir: str):
        # 1) 恢复模型及分类器权重
        state = torch.load(os.path.join(ckpt_dir, WEIGHTS_NAME), map_location='cpu')
        missing, unexpected = self.load_state_dict(state, strict=False)
        logger.info(f'missing keys: {missing}, unexpected keys: {unexpected}')
        # 2) 恢复特征抽取器配置
        self.data_processor.extractor.load(ckpt_dir)
        # 同步 Model 中的列列表
        self.categorical_columns = self.data_processor.extractor.categorical_columns
        self.numerical_columns   = self.data_processor.extractor.numerical_columns
        self.binary_columns      = self.data_processor.extractor.binary_columns

    def update(self, config: dict):
        """
        Update feature columns and optionally rebuild classifier for new num_class.
        config keys: 'cat', 'num', 'bin', 'num_class'
        """
        # 1) 更新列映射
        col_map = {k: v for k, v in config.items() if k in ('cat', 'num', 'bin')}
        if col_map:
            self.data_processor.extractor.update(**col_map)
            self.categorical_columns = self.data_processor.extractor.categorical_columns
            self.numerical_columns   = self.data_processor.extractor.numerical_columns
            self.binary_columns      = self.data_processor.extractor.binary_columns
        # 2) 如需更新分类器输出类别数
        if 'num_class' in config:
            self._adapt_to_new_num_class(config['num_class'])

    def _adapt_to_new_num_class(self, num_class: int):
        """
        When finetuning on a dataset with a different number of classes,
        rebuild the classification head and loss function.
        """
        if not hasattr(self, 'clf') or num_class == getattr(self, 'num_class', None):
            return
        self.num_class = num_class
        # 1) 重建分类头
        self.clf = TransTabLinearClassifier(num_class=num_class,
                                            hidden_dim=self.cls_token.hidden_dim)
        self.clf.to(self.device)
        # 2) 选择合适的损失函数
        if num_class > 2:
            self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        logger.info(f'Rebuilt classifier for num_class={num_class}.')
    '''

class TransTabLinearClassifier(nn.Module):
    """
    简单的线性分类头，直接在 CLS 嵌入上做 LayerNorm + 全连接。
    """
    def __init__(self, num_class: int, hidden_dim: int = 128) -> None:
        super().__init__()
        # 二分类时输出维度 1，多分类时输出 num_class
        out_dim = 1 if num_class <= 2 else num_class
        self.fc = nn.Linear(hidden_dim, out_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, cls_emb: torch.Tensor) -> torch.Tensor:
        """
        cls_emb: [batch, hidden_dim]
        返回 logits: [batch] (二分类) 或 [batch, num_class]
        """
        x = self.norm(cls_emb)
        logits = self.fc(x)
        return logits


class TransTabClassifier(TransTabModel):
    """
    继承自 TransTabModel，在其 CLS 向量之上加一个线性分类头，并计算损失。
    """
    def __init__(
        self,
        categorical_columns: List[str]   = None,
        numerical_columns:   List[str]   = None,
        binary_columns:      List[str]   = None,
        num_class:           int         = 2,
        hidden_dim:          int         = 128,
        num_layer:           int         = 2,
        num_attention_head:  int         = 8,
        hidden_dropout_prob: float       = 0.1,
        ffn_dim:             int         = 256,
        activation:          str         = 'relu',
        device:              Union[str, torch.device] = 'cuda:0',
        **kwargs,
    ) -> None:
        # 调用父类构造器，完成数据提取、DataProcessor、Transformer 等初始化
        super().__init__(
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            binary_columns=binary_columns,
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            num_attention_head=num_attention_head,
            hidden_dropout_prob=hidden_dropout_prob,
            ffn_dim=ffn_dim,
            activation=activation,
            device=device,
            **kwargs,
        )

        self.num_class = num_class
        # 分类头：接收 [batch, hidden_dim] 的 CLS 向量
        self.clf = TransTabLinearClassifier(num_class=num_class, hidden_dim=hidden_dim)

        # 根据类别数选择损失函数
        if num_class > 2:
            self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')

        self.to(self.device)

    def forward(
        self,
        x: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        参数:
          x: pd.DataFrame, 原始表格批量数据
          y: pd.Series 可选标签
        返回:
          logits 或 (logits, loss)
        """
        # 1) 利用父类拿到 CLS 向量 [batch, hidden_dim]
        cls_emb = super().forward(x)

        # 2) 分类头
        logits = self.clf(cls_emb)

        # 3) 如果给了标签，则计算并返回 loss
        if y is not None:
            # 如果 y 已经是 Tensor，直接搬到 device；否则假定它是 pd.Series
            if isinstance(y, torch.Tensor):
                y_ts = y.to(self.device)
            else:
                y_ts = torch.tensor(y.values, device=self.device)

            if self.num_class > 2:
                y_ts = y_ts.long()
                loss = self.loss_fn(logits, y_ts)
            else:
                y_ts = y_ts.float()
                loss = self.loss_fn(logits.view(-1), y_ts)
            loss = loss.mean()
        else:
            loss = None

        return logits, loss


