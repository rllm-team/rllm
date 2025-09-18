from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, Tuple
import math
import os
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

import torch
from torch import Tensor
import numpy as np
import pandas as pd

from rllm.data.table_data import TableData
from rllm.nn.conv.table_conv import TransTabConv
from rllm.nn.pre_encoder import TransTabDataExtractor, TransTabPreEncoder
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


class TransTabCLSToken(torch.nn.Module):
    r"""Add a learnable CLS token embedding to the beginning of each sequence.

    This module maintains a trainable embedding vector of size `hidden_dim`
    that is prepended to the input embeddings on each forward pass. If an
    attention mask is provided, it is also updated to include the new token.

    Args:
        hidden_dim (int): Dimensionality of the CLS token embedding.
    """

    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(Tensor(hidden_dim))
        torch.nn.init.uniform_(self.weight, a=-1 / math.sqrt(hidden_dim), b=1 / math.sqrt(hidden_dim))
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


class TransTabProjectionHead(torch.nn.Module):
    def __init__(self, hidden_dim=128, projection_dim=128):
        super().__init__()
        self.dense = torch.nn.Linear(hidden_dim, projection_dim, bias=False)

    def forward(self, x) -> Tensor:
        h = self.dense(x)
        return h


class TransTab(torch.nn.Module):
    r"""TransTab: End-to-end TransTab model for downstream prediction tasks,
    `"TransTab: Learning Transferable Tabular Transformers Across Tables"`
    <https://arxiv.org/abs/2205.09328>`_ paper.

    This model implements the full TransTab pipeline:
      1) Extract raw DataFrame columns into token IDs and value tensors
         (TransTabDataExtractor).
      2) Convert tokens/values into feature embeddings and attention masks
         (TransTabDataProcessor).
      3) Prepend a learnable [CLS] token and encode the sequence via a
         multi-layer Transformer stack (TransTabCLSToken + TransTabConv).
      4) Produce a single-vector representation from the final CLS position,
         optionally apply a contrastive projection head, and (subclasses may)
         append a classification head.

    Args:
        categorical_columns (Optional[List[str]]): Names of categorical features.
        numerical_columns (Optional[List[str]]): Names of numerical features.
        binary_columns (Optional[List[str]]): Names of binary (0/1) features.
        hidden_dim (int): Dimension of all internal embeddings (d_model).
        num_layer (int): Number of Transformer encoder(conv) layers to stack.
        num_attention_head (int): Number of attention heads per layer.
        hidden_dropout_prob (float): Dropout probability in Transformer sublayers.
        layer_norm_eps (float): Epsilon for all LayerNorm operations.
        ffn_dim (int): Inner dimension of Transformer feedforward networks.
        activation (str): Activation function for feedforward ("relu", etc.).
        device (Union[str, torch.device]): Device on which to run the model.
        projection_dim (int): Output dimension of the contrastive projection head.
        overlap_ratio (float): Overlap fraction used in contrastive partitioning.
        num_partition (int): Number of partitions for contrastive sampling.
        supervised (bool): If True, use supervised contrastive loss; otherwise unsupervised.
        temperature (float): Temperature parameter for contrastive loss.
        base_temperature (float): Base temperature for stability scaling.
        **kwargs: Additional keyword arguments passed to TransTabDataExtractor.
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

        # 2) Initialize DataExtractor
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
            device=device,
            extractor=self.extractor,
            use_align_layer=True,
        )

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            TransTabConv(
                conv_dim=hidden_dim,
                nhead=num_attention_head,
                dropout=hidden_dropout_prob,
                dim_feedforward=ffn_dim,
                activation=activation,
                layer_norm_eps=self.layer_norm_eps,
                norm_first=False,
                use_layer_norm=True,
                batch_first=True,
                device=device,
            )
        )
        for _ in range(num_layer - 1):
            self.convs.append(
                TransTabConv(
                    conv_dim=hidden_dim,
                    nhead=num_attention_head,
                    dropout=hidden_dropout_prob,
                    dim_feedforward=ffn_dim,
                    activation=activation,
                    layer_norm_eps=self.layer_norm_eps,
                    norm_first=False,
                    use_layer_norm=True,
                    batch_first=True,
                    device=device,
                )
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
        self.ce_loss = torch.nn.CrossEntropyLoss()
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
        proc_out = self.pre_encoder(df)
        emb = proc_out['embedding']       # (batch, seq_len, hidden_dim)
        mask = proc_out['attention_mask']  # (batch, seq_len)

        # 2) Add CLS token at the beginning
        cls_out = self.cls_token(emb, attention_mask=mask)
        emb2 = cls_out['embedding']       # (batch, seq_len+1, hidden_dim)
        mask2 = cls_out['attention_mask']  # (batch, seq_len+1)

        # 3) Transformer Encoding
        # enc_out = self.convs(embedding=emb2, attention_mask=mask2)
        for conv in self.convs:
            enc_out = conv(x=emb2, src_key_padding_mask=mask2)
            emb2 = enc_out

        # 4) Take the first position (CLS) as the sample representation
        final_cls = enc_out[:, 0, :]       # (batch, hidden_dim)
        return final_cls

    def save(self, ckpt_dir: str) -> None:
        """
        Save the entire model:
        1) data_processor.save → extractor/tokenizer + pre_encoder
        2) Save the state_dict of this model (conv + cls_token)
        """
        # 1) save extractor + pre_encoder
        self.pre_encoder.save(ckpt_dir)

        # 2) Save model weights
        os.makedirs(ckpt_dir, exist_ok=True)
        model_path = os.path.join(ckpt_dir, WEIGHTS_NAME)
        torch.save(self.state_dict(), model_path)
        logger.info(f"Saved TransTab weights to {model_path}")

    def load(self, ckpt_dir: str) -> None:
        """
        Load the entire model:
        1) Restore extractor/tokenizer + pre_encoder
        2) Synchronize column mapping to model attributes
        3) Load encoder + cls_token's state_dict (to CPU first)
        4) Move the model to self.device
        """
        # 1) Restore extractor + pre_encoder
        self.pre_encoder.load(ckpt_dir)

        # 2) Synchronous column mapping
        self.categorical_columns = self.pre_encoder.extractor.categorical_columns
        self.numerical_columns = self.pre_encoder.extractor.numerical_columns
        self.binary_columns = self.pre_encoder.extractor.binary_columns

        # 3) Load model weights to CPU
        model_path = os.path.join(ckpt_dir, WEIGHTS_NAME)
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded TransTab weights from {model_path}")
        logger.info(f" Missing keys: {missing}")
        logger.info(f" Unexpected keys: {unexpected}")

        # 4) Cache the pre-trained pre_encoder status, which will be used in the subsequent update
        pe_path = os.path.join(ckpt_dir, INPUT_ENCODER_NAME)
        self._preencoder_state = torch.load(
            pe_path,
            map_location='cpu',
            weights_only=True
        )

        self.to(self.device)

    def update(self, config: Dict[str, Any]) -> None:
        col_map = {k: v for k, v in config.items() if k in ('cat', 'num', 'bin')}
        if col_map:
            ext = self.pre_encoder.extractor
            ext.update(
                cat=col_map.get('cat', None),
                num=col_map.get('num', None),
                bin=col_map.get('bin', None),
            )
            self.categorical_columns = ext.categorical_columns
            self.numerical_columns = ext.numerical_columns
            self.binary_columns = ext.binary_columns

            logger.info("Extended column mappings in TransTab via extractor.update().")

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
            self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        else:
            self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
        logger.info(f"Rebuilt classifier for num_class={num_class}.")


class TransTabLinearClassifier(torch.nn.Module):
    r"""TransTabLinearClassifier: Simple linear classification head with
    layer normalization followed by a fully-connected layer on the
    CLS embedding.

    Args:
        num_class (int): Number of target classes (<=2 produces a single logit).
        hidden_dim (int): Dimensionality of input CLS embedding.
    """

    def __init__(self, num_class: int, hidden_dim: int = 128) -> None:
        super().__init__()
        out_dim = 1 if num_class <= 2 else num_class
        self.fc = torch.nn.Linear(hidden_dim, out_dim)
        self.norm = torch.nn.LayerNorm(hidden_dim)

    def forward(self, cls_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cls_emb (torch.Tensor): CLS token embeddings of shape (batch_size, hidden_dim).

        Returns:
            torch.Tensor: Classification logits of shape
                (batch_size,) for binary or (batch_size, num_class).
        """
        x = self.norm(cls_emb)
        logits = self.fc(x)
        return logits


class TransTabLinearRegressor(torch.nn.Module):
    def __init__(self, hidden_dim=128) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(hidden_dim, 1)
        self.norm = torch.nn.LayerNorm(hidden_dim)

    def forward(self, x) -> Tensor:
        x = x[:, 0, :]  # take the cls token embedding
        x = self.norm(x)
        output = self.fc(x)
        return output


class TransTabClassifier(TransTab):
    r"""TransTabClassifier: Classification model built on TransTab.

    Inherits the full TransTab pipeline (data extraction, preprocessing,
    Transformer encoding, CLS token) and adds a linear classification head
    on the final CLS embedding, computing either cross-entropy or binary
    cross-entropy loss.

    Args:
        categorical_columns (Optional[List[str]]): Names of categorical features.
        numerical_columns (Optional[List[str]]): Names of numerical features.
        binary_columns (Optional[List[str]]): Names of binary (0/1) features.
        num_class (int): Number of target classes (≤2 yields a single-logit binary head).
        hidden_dim (int): Dimensionality of internal embeddings (d_model).
        num_layer (int): Number of Transformer encoder layers to stack.
        num_attention_head (int): Number of attention heads per Transformer layer.
        hidden_dropout_prob (float): Dropout probability in Transformer sublayers.
        ffn_dim (int): Inner dimension of Transformer feedforward networks.
        activation (str): Activation function for feedforward layers ("relu", etc.).
        device (Union[str, torch.device]): Device on which to run the model.
        **kwargs: Additional keyword arguments passed to `TransTab`.
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
            self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        else:
            self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')

        self.to(self.device)

    def forward(
        self,
        x: Union[pd.DataFrame, TableData, Dict[ColType, torch.Tensor]],
        y: Optional[Union[pd.Series, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x (DataFrame | TableData | Dict[ColType, Tensor]): Input batch.
            y (Optional[Series | Tensor]): Ground-truth labels.

        Returns:
            Tuple containing:
              - logits (Tensor): Classification logits.
              - loss (Optional[Tensor]): Mean loss if `y` provided, else None.
        """
        cls_emb = super().forward(x)

        logits = self.clf(cls_emb)

        if y is not None:
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


class TransTabForCL(TransTab):
    r"""TransTabForCL: Contrastive learning model subclassing TransTab.

    Implements vertical-partition contrastive learning (VPCL) by sampling
    multiple column subsets per table, encoding each view through the
    TransTab pipeline, projecting to a lower-dimensional space, and
    computing either supervised or self-supervised contrastive loss.

    Args:
        categorical_columns (Optional[List[str]]): Names of categorical features.
        numerical_columns (Optional[List[str]]): Names of numerical features.
        binary_columns (Optional[List[str]]): Names of binary (0/1) features.
        hidden_dim (int): Dimensionality of internal embeddings (d_model).
        num_layer (int): Number of Transformer encoder layers to stack.
        num_attention_head (int): Number of attention heads per layer.
        hidden_dropout_prob (float): Dropout probability in Transformer sublayers.
        ffn_dim (int): Inner dimension of Transformer feedforward networks.
        projection_dim (int): Dimension of the contrastive projection head.
        overlap_ratio (float): Fraction of overlap between adjacent partitions.
        num_partition (int): Number of column partitions per sample.
        supervised (bool): Use supervised contrastive loss if True;
            otherwise self-supervised.
        temperature (float): Temperature scaling for contrastive logits.
        base_temperature (float): Base temperature for loss normalization.
        activation (str): Activation function for feedforward layers.
        device (Union[str, torch.device]): Device on which to run the model.
        **kwargs: Additional keyword arguments passed to `TransTab`.
    """

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
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.num_partition = num_partition
        self.overlap_ratio = overlap_ratio
        self.supervised = supervised
        self.device = device
        self.to(device)

    def forward(self, x, y=None):
        """
        Args:
            x (DataFrame | Dict[ColType, Tensor]): Input batch or extracted features.
            y (Optional[Tensor]): Labels for supervised contrastive loss.

        Returns:
            Tuple containing:
              - None (no logits returned for CL model)
              - loss (Tensor): Computed contrastive loss.
        """
        # do positive sampling
        feat_x_list = []
        if isinstance(x, pd.DataFrame):
            sub_x_list = self._build_positive_pairs(x, self.num_partition)
            for sub_x in sub_x_list:
                proc = self.data_processor(sub_x)
                emb = proc["embedding"]
                mask = proc["attention_mask"]

                cls_out = self.cls_token(emb, attention_mask=mask)
                enc_out = self.convs(
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
        r"""Compute self‑supervised vertical‑partition contrastive loss (VPCL).

        Treats each column partition as a separate view and maximizes agreement
        between partitions of the same sample without using labels.

        Args:
            features (torch.Tensor): Encoded features of shape
                `(batch_size, num_partition, proj_dim)`.

        Returns:
            torch.Tensor: Scalar self‑supervised contrastive loss.
        """
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
        r"""Compute supervised vertical‑partition contrastive loss (VPCL).

        Uses class labels to form positive pairs across partitions, encouraging
        same‑class views to be closer in projection space.

        Args:
            features (torch.Tensor): Encoded features of shape
                `(batch_size, num_partition, proj_dim)`.
            labels (torch.Tensor): Ground‑truth labels of shape `(batch_size,)`.

        Returns:
            torch.Tensor: Scalar supervised contrastive loss.
        """
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
