from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, Tuple

import math
import os

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.init as nn_init
import numpy as np
import pandas as pd
from loguru import logger

from rllm.data.table_data import TableData
from rllm.nn.conv.table_conv import (
    constants,
    TransTabConv,
    TransTabDataExtractor,
    TransTabDataProcessor,
)
from rllm.nn.pre_encoder import TransTabPreEncoder
from rllm.types import ColType


class TransTabCLSToken(nn.Module):
    '''add a learnable cls token embedding at the end of each sequence.
    '''
    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.weight = nn.Parameter(Tensor(hidden_dim))
        nn_init.uniform_(self.weight, a=-1 / math.sqrt(hidden_dim), b=1 / math.sqrt(hidden_dim))
        self.hidden_dim = hidden_dim

    def expand(self, *leading_dimensions):
        new_dims = (1,) * (len(leading_dimensions) - 1)
        return self.weight.view(*new_dims, -1).expand(*leading_dimensions, -1)

    def forward(self, embedding, attention_mask=None, **kwargs) -> Tensor:
        embedding = torch.cat([self.expand(len(embedding), 1), embedding], dim=1)
        outputs = {'embedding': embedding}
        if attention_mask is not None:
            attention_mask = torch.cat(
                [torch.ones(attention_mask.shape[0], 1).to(attention_mask.device), attention_mask], 1)
        outputs['attention_mask'] = attention_mask
        return outputs


class TransTabProjectionHead(nn.Module):
    def __init__(self, hidden_dim=128, projection_dim=128):
        super().__init__()
        self.dense = nn.Linear(hidden_dim, projection_dim, bias=False)

    def forward(self, x) -> Tensor:
        h = self.dense(x)
        return h


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
        categorical_columns: List[str] = None,
        numerical_columns: List[str] = None,
        binary_columns: List[str] = None,
        hidden_dim: int = 128,
        num_layer: int = 2,
        num_attention_head: int = 8,
        hidden_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-5,
        ffn_dim: int = 256,
        activation: str = 'relu',
        device: Union[str, torch.device] = 'cuda:0',
        projection_dim: int = 128,
        overlap_ratio: float = 0.1,
        num_partition: int = 2,
        supervised: bool = True,
        temperature: float = 10.0,
        base_temperature: float = 10.0,
        **kwargs,
    ) -> None:
        super().__init__()

        self.hidden_dropout_prob = hidden_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        # 1) Record and deduplicate various column names
        self.categorical_columns = list(set(categorical_columns)) if categorical_columns else None
        self.numerical_columns = list(set(numerical_columns)) if numerical_columns else None
        self.binary_columns = list(set(binary_columns)) if binary_columns else None

        # 2) Initialize DataExtractor (keep all **kwargs configuration)
        self.extractor = TransTabDataExtractor(
            categorical_columns=self.categorical_columns,
            numerical_columns=self.numerical_columns,
            binary_columns=self.binary_columns,
            **kwargs,
        )

        # 3) Initialize PreEncoder (for DataProcessor), metadata provides an empty list mapping
        metadata = {
            ColType.CATEGORICAL: [],
            ColType.BINARY: [],
            ColType.NUMERICAL: [],
        }
        self.pre_encoder = TransTabPreEncoder(
            out_dim=hidden_dim,
            metadata=metadata,
            vocab_size=self.extractor.tokenizer.vocab_size,
            padding_idx=self.extractor.tokenizer.pad_token_id,
            hidden_dropout_prob=self.hidden_dropout_prob,
            layer_norm_eps=self.layer_norm_eps,
        )

        # 4) Initialize DataProcessor and combine extractor with pre_encoder
        self.data_processor = TransTabDataProcessor(
            pre_encoder=self.pre_encoder,
            out_dim=hidden_dim,
            device=device,
        )

        # 5) Building a multi-layer Transformer encoder (using TransTabConv)
        self.encoder = TransTabConv(
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            num_attention_head=num_attention_head,
            hidden_dropout_prob=hidden_dropout_prob,
            ffn_dim=ffn_dim,
            activation=activation,
            layer_norm_eps=self.layer_norm_eps,
            norm_first=False,
            use_layer_norm=True,
            batch_first=True,
            device=device,
        )

        # 6) CLS token module, used to insert a learnable vector at the front of the sequence
        self.cls_token = TransTabCLSToken(hidden_dim=hidden_dim)

        self.device = device
        self.to(device)

        # Contrastive Learning
        # Add a small projection head on top of the CLS embedding
        self.projection_head = TransTabProjectionHead(hidden_dim, projection_dim)
        # CL hyperparameters
        self.supervised = supervised
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.num_partition = num_partition
        self.overlap_ratio = overlap_ratio
        self.ce_loss = nn.CrossEntropyLoss()
        # device already set

    def forward(
        self,
        x: Union[pd.DataFrame, TableData, Dict[ColType, torch.Tensor]],
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: a batch of samples
        y: optional label (placeholder only, ignored by the base class)
        Return: final [CLS] vector, shape = (batch, hidden_dim)
        """
        if isinstance(x, dict):
            import pandas as pd
            parts = []
            if x.get(ColType.CATEGORICAL) is not None:
                parts.append(
                    pd.DataFrame(
                        x[ColType.CATEGORICAL].cpu().numpy(),
                        columns=self.categorical_columns,
                    )
                )
            if x.get(ColType.NUMERICAL) is not None:
                parts.append(
                    pd.DataFrame(
                        x[ColType.NUMERICAL].cpu().numpy(),
                        columns=self.numerical_columns,
                    )
                )
            if x.get(ColType.BINARY) is not None:
                parts.append(
                    pd.DataFrame(
                        x[ColType.BINARY].cpu().numpy(),
                        columns=self.binary_columns,
                    )
                )
            df = pd.concat(parts, axis=1)
        elif isinstance(x, TableData):
            df = x.df
        else:
            df = x

        # 1) DataProcessor gets embedding + mask
        proc_out = self.data_processor(df)
        emb = proc_out['embedding']       # (batch, seq_len, hidden_dim)
        mask = proc_out['attention_mask']  # (batch, seq_len)

        # 2) Add CLS token at the beginning
        cls_out = self.cls_token(emb, attention_mask=mask)
        emb2 = cls_out['embedding']       # (batch, seq_len+1, hidden_dim)
        mask2 = cls_out['attention_mask']  # (batch, seq_len+1)

        # 3) Transformer Encoding
        enc_out = self.encoder(embedding=emb2, attention_mask=mask2)

        # 4) Take the first position (CLS) as the sample representation
        final_cls = enc_out[:, 0, :]       # (batch, hidden_dim)
        return final_cls

    def save(self, ckpt_dir: str) -> None:
        """
        Save the entire model:
        1) data_processor.save → extractor/tokenizer + pre_encoder
        2) Save the state_dict of this model (encoder + cls_token)
        """
        # 1) save extractor + pre_encoder
        self.data_processor.save(ckpt_dir)

        # 2) Save model weights
        os.makedirs(ckpt_dir, exist_ok=True)
        model_path = os.path.join(ckpt_dir, constants.WEIGHTS_NAME)
        torch.save(self.state_dict(), model_path)
        logger.info(f"Saved TransTabModel weights to {model_path}")

    def load(self, ckpt_dir: str) -> None:
        """
        Load the entire model:
        1) Restore extractor/tokenizer + pre_encoder
        2) Synchronize column mapping to model attributes
        3) Load encoder + cls_token's state_dict (to CPU first)
        4) Move the model to self.device
        """
        # 1) Restore extractor + pre_encoder
        self.data_processor.load(ckpt_dir)

        # 2) Synchronous column mapping
        self.categorical_columns = self.data_processor.extractor.categorical_columns
        self.numerical_columns = self.data_processor.extractor.numerical_columns
        self.binary_columns = self.data_processor.extractor.binary_columns

        # 3) Load model weights to CPU
        model_path = os.path.join(ckpt_dir, constants.WEIGHTS_NAME)
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded TransTabModel weights from {model_path}")
        logger.info(f" Missing keys: {missing}")
        logger.info(f" Unexpected keys: {unexpected}")

        # 4) Cache the pre-trained pre_encoder status, which will be used in the subsequent update
        pe_path = os.path.join(ckpt_dir, constants.INPUT_ENCODER_NAME)
        self._preencoder_state = torch.load(
            pe_path,
            map_location='cpu',
            weights_only=True
        )

        self.to(self.device)

    def update(self, config: Dict[str, Any]) -> None:
        # Completely replace the extractor column
        col_map = {k: v for k, v in config.items() if k in ('cat', 'num', 'bin')}
        if col_map:
            ext = self.data_processor.extractor
            if 'cat' in col_map:
                ext.categorical_columns = list(col_map['cat'])
            if 'num' in col_map:
                ext.numerical_columns = list(col_map['num'])
            if 'bin' in col_map:
                ext.binary_columns = list(col_map['bin'])

            ok, dup = ext._check_column_overlap(
                ext.categorical_columns,
                ext.numerical_columns,
                ext.binary_columns)
            if not ok:
                if not ext.ignore_duplicate_cols:
                    raise ValueError(f"Column overlap after update: {dup}")
                ext._solve_duplicate_cols(dup)

            # Synchronize top-level properties
            self.categorical_columns = ext.categorical_columns
            self.numerical_columns = ext.numerical_columns
            self.binary_columns = ext.binary_columns
            logger.info("Updated column mappings in TransTabModel.")

            # 2) Rebuild pre_encoder + reload embedding
            new_meta = {
                ColType.CATEGORICAL: self.categorical_columns,
                ColType.BINARY: self.binary_columns,
                ColType.NUMERICAL: self.numerical_columns,
            }
            self.pre_encoder = TransTabPreEncoder(
                out_dim=self.cls_token.hidden_dim,
                metadata=new_meta,
                vocab_size=ext.tokenizer.vocab_size,
                padding_idx=ext.tokenizer.pad_token_id,
                hidden_dropout_prob=self.hidden_dropout_prob,
                layer_norm_eps=self.layer_norm_eps,
            ).to(self.device)
            # Reload the cached weights from pre-training
            self.pre_encoder.load_state_dict(self._preencoder_state, strict=False)

            # Rebuild data_processor
            self.data_processor = TransTabDataProcessor(
                pre_encoder=self.pre_encoder,
                out_dim=self.cls_token.hidden_dim,
                device=self.device,
            )

        # 3) Rebuild the classification header (optional)
        if 'num_class' in config:
            self._adapt_to_new_num_class(config['num_class'])

    def _adapt_to_new_num_class(self, num_class: int) -> None:
        """
        If this model (or a subclass) defines self.clf,
        rebuild the classification head and loss_fn when the number of classes changes.
        """
        if not hasattr(self, 'clf') or num_class == getattr(self, 'num_class', None):
            return
        self.num_class = num_class
        # Rebuilding the classification header
        self.clf = TransTabLinearClassifier(num_class=num_class,
                                            hidden_dim=self.cls_token.hidden_dim)
        self.clf.to(self.device)
        # Reconstruction loss
        if num_class > 2:
            self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        logger.info(f"Rebuilt classifier for num_class={num_class}.")


class TransTabLinearClassifier(nn.Module):
    """
    Simple linear classification head, LayerNorm + fully connected directly on CLS embedding.
    """
    def __init__(self, num_class: int, hidden_dim: int = 128) -> None:
        super().__init__()
        out_dim = 1 if num_class <= 2 else num_class
        self.fc = nn.Linear(hidden_dim, out_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, cls_emb: torch.Tensor) -> torch.Tensor:
        """
        cls_emb: [batch, hidden_dim]
        Returns logits: [batch] (binary classification) or [batch, num_class]
        """
        x = self.norm(cls_emb)
        logits = self.fc(x)
        return logits


class TransTabClassifier(TransTabModel):
    """
    Inherits from TransTabModel, adds a linear classification head on top of its CLS vector, and calculates the loss.
    """
    def __init__(
        self,
        categorical_columns: List[str] = None,
        numerical_columns: List[str] = None,
        binary_columns: List[str] = None,
        num_class: int = 2,
        hidden_dim: int = 128,
        num_layer: int = 2,
        num_attention_head: int = 8,
        hidden_dropout_prob: float = 0.1,
        ffn_dim: int = 256,
        activation: str = 'relu',
        device: Union[str, torch.device] = 'cuda:0',
        **kwargs,
    ) -> None:
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
        # Classification head: receives a CLS vector of [batch, hidden_dim]
        self.clf = TransTabLinearClassifier(num_class=num_class, hidden_dim=hidden_dim)

        if num_class > 2:
            self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')

        self.to(self.device)

    def forward(
        self,
        x: Union[pd.DataFrame, TableData, Dict[ColType, torch.Tensor]],
        y: Optional[Union[pd.Series, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Parameters:
        x: pd.DataFrame, original table batch data
        y: pd.Series optional labels
        Return:
        logits or (logits, loss)
        """
        # Use the parent class to get the CLS vector [batch, hidden_dim]
        cls_emb = super().forward(x)

        logits = self.clf(cls_emb)

        # If a label is given, calculate and return the loss
        if y is not None:
            # If y is already a Tensor, move it directly to the device; otherwise it is assumed to be a pd.Series
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


class TransTabForCL(TransTabModel):
    '''The contrasstive learning model subclass from :class:`transtab.modeling_transtab.TransTabModel`.

    Parameters
    ----------
    categorical_columns: list
        a list of categorical feature names.

    numerical_columns: list
        a list of numerical feature names.

    binary_columns: list
        a list of binary feature names, accept binary indicators like (yes,no); (true,false); (0,1).

    feature_extractor: TransTabFeatureExtractor
        a feature extractor to tokenize the input tables. if not passed the model will build itself.

    hidden_dim: int
        the dimension of hidden embeddings.

    num_layer: int
        the number of transformer layers used in the encoder.

    num_attention_head: int
        the numebr of heads of multihead self-attention layer in the transformers.

    hidden_dropout_prob: float
        the dropout ratio in the transformer encoder.

    ffn_dim: int
        the dimension of feed-forward layer in the transformer layer.

    projection_dim: int
        the dimension of projection head on the top of encoder.

    overlap_ratio: float
        the overlap ratio of columns of different partitions when doing subsetting.

    num_partition: int
        the number of partitions made for vertical-partition contrastive learning.

    supervised: bool
        whether or not to take supervised VPCL, otherwise take self-supervised VPCL.

    temperature: float
        temperature used to compute logits for contrastive learning.

    base_temperature: float
        base temperature used to normalize the temperature.

    activation: str
        the name of used activation functions, support ``"relu"``, ``"gelu"``, ``"selu"``, ``"leakyrelu"``.

    device: str
        the device, ``"cpu"`` or ``"cuda:0"``.

    Returns
    -------
    A TransTabForCL model.

    '''
    def __init__(
        self,
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        hidden_dim=128,
        num_layer=2,
        num_attention_head=8,
        hidden_dropout_prob=0,
        ffn_dim=256,
        projection_dim=128,
        overlap_ratio=0.1,
        num_partition=2,
        supervised=True,
        temperature=10,
        base_temperature=10,
        activation='relu',
        device='cuda:0',
        **kwargs,
    ) -> None:
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
        assert num_partition > 0, f'number of contrastive subsets must be greater than 0, got {num_partition}'
        assert isinstance(num_partition, int), f'number of constrative subsets must be int, got {type(num_partition)}'
        assert overlap_ratio >= 0 and overlap_ratio < 1, f'overlap_ratio must be in [0, 1), got {overlap_ratio}'
        self.projection_head = TransTabProjectionHead(hidden_dim, projection_dim)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.num_partition = num_partition
        self.overlap_ratio = overlap_ratio
        self.supervised = supervised
        self.device = device
        self.to(device)

    def forward(self, x, y=None):
        '''Make forward pass given the input feature ``x`` and label ``y`` (optional).

        Parameters
        ----------
        x: pd.DataFrame or dict
            pd.DataFrame: a batch of raw tabular samples; dict: the output of TransTabFeatureExtractor.

        y: pd.Series
            the corresponding labels for each sample in ``x``. if label is given, the model will return
            the classification loss by ``self.loss_fn``.

        Returns
        -------
        logits: None
            this CL model does NOT return logits.

        loss: torch.Tensor
            the supervised or self-supervised VPCL loss.

        '''
        # do positive sampling
        feat_x_list = []
        if isinstance(x, pd.DataFrame):
            sub_x_list = self._build_positive_pairs(x, self.num_partition)
            for sub_x in sub_x_list:
                proc = self.data_processor(sub_x)
                emb = proc["embedding"]
                mask = proc["attention_mask"]

                cls_out = self.cls_token(emb, attention_mask=mask)
                enc_out = self.encoder(
                    embedding=cls_out["embedding"],
                    attention_mask=cls_out["attention_mask"],
                )

                feat = enc_out[:, 0, :]      # [bs, hidden_dim]
                feat_proj = self.projection_head(feat)  # [bs, proj_dim]
                feat_x_list.append(feat_proj)
        else:
            raise ValueError(f"Expected input type pd.DataFrame, got {type(x)}")

        # bs, num_partition, proj_dim
        feat_x_multiview = torch.stack(feat_x_list, dim=1)

        if y is not None and self.supervised:
            labels = y.to(self.device).long()
            loss = self.supervised_contrastive_loss(feat_x_multiview, labels)
        else:
            # print("#0628##########No labels provided, using self-supervised contrastive loss.##############")
            loss = self.self_supervised_contrastive_loss(feat_x_multiview)

        return None, loss

    def _build_positive_pairs(self, x, n):
        x_cols = x.columns.tolist()
        sub_col_list = np.array_split(np.array(x_cols), n)
        len_cols = len(sub_col_list[0])
        overlap = int(np.ceil(len_cols * (self.overlap_ratio)))
        sub_x_list = []
        for i, sub_col in enumerate(sub_col_list):
            if overlap > 0 and i < n - 1:
                sub_col = np.concatenate([sub_col, sub_col_list[i + 1][:overlap]])
            elif overlap > 0 and i == n - 1:
                sub_col = np.concatenate([sub_col, sub_col_list[i - 1][-overlap:]])
            sub_x = x.copy()[sub_col]
            sub_x_list.append(sub_x)
        return sub_x_list

    def cos_sim(self, a, b):
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a_norm, b_norm.transpose(0, 1))

    def self_supervised_contrastive_loss(self, features):
        '''Compute the self-supervised VPCL loss.

        Parameters
        ----------
        features: torch.Tensor
            the encoded features of multiple partitions of input tables, with shape ``(bs, n_partition, proj_dim)``.

        Returns
        -------
        loss: torch.Tensor
            the computed self-supervised VPCL loss.
        '''
        batch_size = features.shape[0]
        labels = torch.arange(batch_size, dtype=torch.long, device=self.device).view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(labels.device)

        contrast_count = features.shape[1]
        # [[0,1],[2,3]] -> [0,2,1,3]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        mask = mask.repeat(anchor_count, contrast_count)
        indices_for_scatter = (
            torch.arange(batch_size * anchor_count)
            .view(-1, 1)
            .to(features.device)
        )
        logits_mask = torch.scatter(torch.ones_like(mask), 1, indices_for_scatter, 0)
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss

    def supervised_contrastive_loss(self, features, labels):
        '''Compute the supervised VPCL loss.

        Parameters
        ----------
        features: torch.Tensor
            the encoded features of multiple partitions of input tables, with shape ``(bs, n_partition, proj_dim)``.

        labels: torch.Tensor
            the class labels to be used for building positive/negative pairs in VPCL.

        Returns
        -------
        loss: torch.Tensor
            the computed VPCL loss.

        '''
        labels = labels.contiguous().view(-1, 1)
        batch_size = features.shape[0]
        mask = torch.eq(labels, labels.T).float().to(labels.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        # contrast_mode == 'all'
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(features.device),
            0,
        )
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss
