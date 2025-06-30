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


