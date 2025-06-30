import torch
import torch.nn as nn
import torch.nn.init as nn_init
from torch import Tensor
import math
import numpy as np
import os
import pandas as pd
from transformers import BertTokenizerFast
from typing import Any, Dict, List
from rllm.types import ColType


class TransTabWordEmbedding(nn.Module):
    r'''
    Encode tokens drawn from column names, categorical and binary features.
    '''
    def __init__(self, vocab_size, hidden_dim, padding_idx=0, hidden_dropout_prob=0, layer_norm_eps=1e-5) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        nn_init.kaiming_normal_(self.word_embeddings.weight)
        self.norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids) -> Tensor:
        embeddings = self.word_embeddings(input_ids)
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class TransTabNumEmbedding(nn.Module):
    r'''
    Encode tokens drawn from column names and the corresponding numerical features.
    '''
    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.num_bias = nn.Parameter(Tensor(1, 1, hidden_dim))  # add bias
        nn_init.uniform_(self.num_bias, a=-1/math.sqrt(hidden_dim), b=1/math.sqrt(hidden_dim))

    def forward(self, num_col_emb, x_num_ts, num_mask=None) -> Tensor:
        num_col_emb = num_col_emb.unsqueeze(0).expand((x_num_ts.shape[0], -1, -1))
        num_feat_emb = num_col_emb * x_num_ts.unsqueeze(-1).float() + self.num_bias
        return num_feat_emb

class TransTabFeatureExtractor:
    r'''
    Process input dataframe to input indices towards transtab encoder.
    '''
    def __init__(self, categorical_columns=None, numerical_columns=None, binary_columns=None, **kwargs) -> None:
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_token_id = self.tokenizer.pad_token_id

        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.binary_columns = binary_columns

    def __call__(self, x) -> Dict:
        encoded_inputs = {
            'x_num': None,
            'num_col_input_ids': None,
            'x_cat_input_ids': None,
            'x_bin_input_ids': None,
        }
        col_names = x.columns.tolist()
        cat_cols = [c for c in col_names if c in self.categorical_columns] if self.categorical_columns is not None else []
        num_cols = [c for c in col_names if c in self.numerical_columns] if self.numerical_columns is not None else []
        bin_cols = [c for c in col_names if c in self.binary_columns] if self.binary_columns is not None else []

        if len(cat_cols+num_cols+bin_cols) == 0:
            cat_cols = col_names

        # Handle NaN for numerical columns
        if len(num_cols) > 0:
            x_num = x[num_cols]
            x_num = x_num.fillna(0)  # fill NaN with zero
            x_num_ts = torch.tensor(x_num.values, dtype=float)
            num_col_ts = self.tokenizer(num_cols, padding=True, truncation=True, add_special_tokens=False, return_tensors='pt')
            encoded_inputs['x_num'] = x_num_ts
            encoded_inputs['num_col_input_ids'] = num_col_ts['input_ids']
            encoded_inputs['num_att_mask'] = num_col_ts['attention_mask']

        # Process categorical columns
        if len(cat_cols) > 0:
            x_cat = x[cat_cols].astype(str)
            x_mask = (~pd.isna(x_cat)).astype(int)
            x_cat = x_cat.fillna('')
            x_cat = x_cat.apply(lambda x: x.name + ' ' + x) * x_mask  # mask out NaN features
            x_cat_str = x_cat.agg(' '.join, axis=1).values.tolist()
            x_cat_ts = self.tokenizer(x_cat_str, padding=True, truncation=True, add_special_tokens=False, return_tensors='pt')

            encoded_inputs['x_cat_input_ids'] = x_cat_ts['input_ids']
            encoded_inputs['cat_att_mask'] = x_cat_ts['attention_mask']

        # Process binary columns
        if len(bin_cols) > 0:
            x_bin = x[bin_cols]
            x_bin_str = x_bin.apply(lambda x: x.name + ' ') * x_bin
            x_bin_str = x_bin_str.agg(' '.join, axis=1).values.tolist()
            x_bin_ts = self.tokenizer(x_bin_str, padding=True, truncation=True, add_special_tokens=False, return_tensors='pt')
            encoded_inputs['x_bin_input_ids'] = x_bin_ts['input_ids']
            encoded_inputs['bin_att_mask'] = x_bin_ts['attention_mask']

        return encoded_inputs

class TransTabFeatureProcessor(nn.Module):
    r'''
    Process inputs from feature extractor to map them to embeddings.
    '''
    def __init__(self, vocab_size=None, hidden_dim=128, device='cuda:0') -> None:
        super().__init__()
        self.word_embedding = TransTabWordEmbedding(vocab_size, hidden_dim)
        self.num_embedding = TransTabNumEmbedding(hidden_dim)
        self.align_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.device = device

    def _avg_embedding_by_mask(self, embs, att_mask=None):
        if att_mask is None:
            return embs.mean(1)
        else:
            embs[att_mask == 0] = 0
            embs = embs.sum(1) / att_mask.sum(1, keepdim=True).to(embs.device)
            return embs

    def forward(self, x_num=None, num_col_input_ids=None, num_att_mask=None,
                x_cat_input_ids=None, cat_att_mask=None, x_bin_input_ids=None, bin_att_mask=None, **kwargs) -> Tensor:
        num_feat_embedding = None
        cat_feat_embedding = None
        bin_feat_embedding = None

        # Process numerical features
        if x_num is not None and num_col_input_ids is not None:
            num_col_emb = self.word_embedding(num_col_input_ids.to(self.device)) 
            x_num = x_num.to(self.device)
            num_col_emb = self._avg_embedding_by_mask(num_col_emb, num_att_mask)
            num_feat_embedding = self.num_embedding(num_col_emb, x_num)
            num_feat_embedding = self.align_layer(num_feat_embedding)

        # Process categorical features
        if x_cat_input_ids is not None:
            cat_feat_embedding = self.word_embedding(x_cat_input_ids.to(self.device))
            cat_feat_embedding = self.align_layer(cat_feat_embedding)

        # Process binary features
        if x_bin_input_ids is not None:
            if x_bin_input_ids.shape[1] == 0: 
                x_bin_input_ids = torch.zeros(x_bin_input_ids.shape[0], dtype=int)[:, None]
            bin_feat_embedding = self.word_embedding(x_bin_input_ids.to(self.device))
            bin_feat_embedding = self.align_layer(bin_feat_embedding)

        # Concatenate all embeddings
        emb_list = []
        att_mask_list = []
        if num_feat_embedding is not None:
            emb_list += [num_feat_embedding]
            att_mask_list += [torch.ones(num_feat_embedding.shape[0], num_feat_embedding.shape[1])]
        if cat_feat_embedding is not None:
            emb_list += [cat_feat_embedding]
            att_mask_list += [cat_att_mask]
        if bin_feat_embedding is not None:
            emb_list += [bin_feat_embedding]
            att_mask_list += [bin_att_mask]
        
        if len(emb_list) == 0:
            raise Exception('No features found for numerical, categorical, or binary types. Please check your data!')
        
        all_feat_embedding = torch.cat(emb_list, 1).float()
        attention_mask = torch.cat(att_mask_list, 1).to(all_feat_embedding.device)
        return {'embedding': all_feat_embedding, 'attention_mask': attention_mask}

class TransTabPreEncoder(nn.Module):
    r"""The TransTabPreEncoder class processes categorical, numerical, and binary features into embeddings for the TransTab model."""

    def __init__(self, out_dim: int, metadata: Dict[ColType, List[Dict[str, Any]]], vocab_size: int, hidden_dim: int, device: str = 'cuda:0') -> None:
        super().__init__()
        self.feature_extractor = TransTabFeatureExtractor(**metadata)
        self.feature_processor = TransTabFeatureProcessor(vocab_size, hidden_dim, device=device)

    def forward(self, feat_dict: Dict[ColType, torch.Tensor], return_dict: bool = False):
        tokenized = self.feature_extractor(feat_dict)
        embeds = self.feature_processor(**tokenized)
        return embeds

#TODO 需要检查pre_encoder的功能是否符合定义，以及是否承接Table data的数据格式。
