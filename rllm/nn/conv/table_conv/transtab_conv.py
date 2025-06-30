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
        self.categorical_columns = categorical_columns
        self.numerical_columns   = numerical_columns
        self.binary_columns      = binary_columns

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
            x_num_df = df[num_cols].fillna(0)
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
            return logits, loss

        return logits


