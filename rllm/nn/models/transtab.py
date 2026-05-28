from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, Tuple
import math
import os

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from rllm.types import ColType
from rllm.preprocessing import TokenizerConfig
from rllm.data.table_data import TableData
from rllm.nn.loss import SupervisedVPCL, SelfSupervisedVPCL
from rllm.nn.encoder import TransTabPreEncoder
from rllm.nn.conv.table_conv import TransTabConv


class TransTabCLSToken(torch.nn.Module):
    r"""Prepends a learnable ``[CLS]`` token to each input sequence.

    Args:
        hidden_dim (int): Dimensionality of the ``[CLS]`` embedding.

    Shape:
        - Input: :math:`(N, S, H)` → Output: :math:`(N, S+1, H)`

    Examples::

        >>> cls = TransTabCLSToken(hidden_dim=8)
        >>> out = cls(torch.randn(2, 5, 8))
        >>> out["embedding"].shape
        torch.Size([2, 6, 8])
    """

    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.weight = torch.nn.Parameter(Tensor(self.hidden_dim))
        torch.nn.init.uniform_(
            self.weight,
            a=-1 / math.sqrt(self.hidden_dim),
            b=1 / math.sqrt(self.hidden_dim),
        )

    def expand(self, *leading_dimensions):
        new_dims = (1,) * (len(leading_dimensions) - 1)
        return self.weight.view(*new_dims, -1).expand(*leading_dimensions, -1)

    def forward(self, embedding, attention_mask=None, **kwargs) -> Tensor:
        embedding = torch.cat([self.expand(len(embedding), 1), embedding], dim=1)
        outputs = {"embedding": embedding}
        if attention_mask is not None:
            attention_mask = torch.cat(
                [
                    torch.ones(attention_mask.shape[0], 1).to(attention_mask.device),
                    attention_mask,
                ],
                1,
            )
        outputs["attention_mask"] = attention_mask
        return outputs


class TransTab(torch.nn.Module):
    r"""Base TransTab encoder for tabular data
    (`"TransTab" <https://arxiv.org/abs/2205.09328>`_).

    Encodes column names and cell values into token embeddings, prepends a
    learnable ``[CLS]`` token, and refines the sequence through ``num_layer``
    Transformer layers.  The final ``[CLS]`` position is returned as a
    fixed-size table-level embedding.

    Args:
        categorical_columns (List[str], optional): Categorical column names. Default: ``None``.
        numerical_columns (List[str], optional): Numerical column names. Default: ``None``.
        binary_columns (List[str], optional): Binary column names. Default: ``None``.
        hidden_dim (int): Shared embedding dimensionality. Default: ``128``.
        num_layer (int): Number of Transformer layers. Default: ``2``.
        num_attention_head (int): Number of attention heads. Default: ``8``.
        hidden_dropout_prob (float): Dropout probability. Default: ``0.1``.
        layer_norm_eps (float): LayerNorm :math:`\varepsilon`. Default: ``1e-5``.
        ffn_dim (int): Feedforward inner dimension. Default: ``256``.
        activation (str): Feedforward activation. Default: ``"relu"``.
        tokenizer: Pre-trained tokenizer; created automatically when ``None``. Default: ``None``.
        **kwargs: Forwarded to :class:`~rllm.nn.encoder.TransTabPreEncoder`.

    Examples::

        >>> from rllm.nn.models import TransTab
        >>> model = TransTab(hidden_dim=32, num_layer=1, num_attention_head=4)
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
        activation: str = "relu",
        projection_dim: int = 128,
        overlap_ratio: float = 0.1,
        num_partition: int = 2,
        supervised: bool = True,
        temperature: float = 10.0,
        base_temperature: float = 10.0,
        tokenizer=None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.hidden_dropout_prob = hidden_dropout_prob
        self.layer_norm_eps = layer_norm_eps

        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.binary_columns = binary_columns

        metadata = {
            ColType.CATEGORICAL: [],
            ColType.BINARY: [],
            ColType.NUMERICAL: [],
        }
        self.pre_encoder = TransTabPreEncoder(
            out_dim=hidden_dim,
            metadata=metadata,
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            binary_columns=binary_columns,
            tokenizer=tokenizer,
            hidden_dropout_prob=self.hidden_dropout_prob,
            layer_norm_eps=self.layer_norm_eps,
            use_align_layer=True,
            **kwargs,
        )

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layer):
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
                )
            )

        self.cls_token = TransTabCLSToken(hidden_dim=hidden_dim)

        self.projection_head = torch.nn.Linear(hidden_dim, projection_dim, bias=False)
        self.supervised = supervised
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.num_partition = num_partition
        self.overlap_ratio = overlap_ratio
        self.ce_loss = torch.nn.CrossEntropyLoss()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        x: Union[pd.DataFrame, TableData, Dict[ColType, torch.Tensor]],
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Encode a table batch into a ``[CLS]`` embedding of shape :math:`(N, H)`.

        Args:
            x (TableData): Input table batch.
            y (Tensor, optional): Unused; kept for subclass API symmetry. Default: ``None``.

        Returns:
            Tensor: ``[CLS]`` embeddings of shape :math:`(N, H)`.
        """
        if isinstance(x, TableData) or hasattr(x, "feat_dict"):
            proc_out = self.pre_encoder(x)
        else:
            raise ValueError(
                f"Expected input type TableData or object with feat_dict, got {type(x)}"
            )

        emb = proc_out["embedding"]  # (batch, seq_len, hidden_dim)
        mask = proc_out["attention_mask"]  # (batch, seq_len)

        # Prepend CLS token
        cls_out = self.cls_token(emb, attention_mask=mask)
        emb2 = cls_out["embedding"]  # (batch, seq_len+1, hidden_dim)
        mask2 = cls_out["attention_mask"]  # (batch, seq_len+1)

        # Transformer encoding
        for conv in self.convs:
            enc_out = conv(x=emb2, src_key_padding_mask=mask2)
            emb2 = enc_out

        final_cls = enc_out[:, 0, :]  # (batch, hidden_dim)
        return final_cls

    def save(self, ckpt_dir: str) -> None:
        r"""Save model weights to ``ckpt_dir``."""
        self.pre_encoder.save(ckpt_dir)

        os.makedirs(ckpt_dir, exist_ok=True)
        model_path = os.path.join(ckpt_dir, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        print(f"Saved TransTab weights to {model_path}")

    def load(self, ckpt_dir: str) -> None:
        r"""Restore model weights from ``ckpt_dir`` (``strict=False``)."""
        self.pre_encoder.load(ckpt_dir)

        # Synchronize column mapping
        self.categorical_columns = self.pre_encoder.categorical_columns
        self.numerical_columns = self.pre_encoder.numerical_columns
        self.binary_columns = self.pre_encoder.binary_columns

        model_path = os.path.join(ckpt_dir, "pytorch_model.bin")
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"Loaded TransTab weights from {model_path}")
        print(f" Missing keys: {missing}")
        print(f" Unexpected keys: {unexpected}")

        # Cache the pre-trained table_encoder state for subsequent update calls
        pe_path = os.path.join(ckpt_dir, "input_encoder.bin")
        self._table_encoder_state = torch.load(
            pe_path, map_location="cpu", weights_only=True
        )

    def update(self, config: Dict[str, Any]) -> None:
        col_map = {k: v for k, v in config.items() if k in ("cat", "num", "bin")}
        if col_map:
            self.pre_encoder.update(
                cat=col_map.get("cat", None),
                num=col_map.get("num", None),
                bin=col_map.get("bin", None),
            )
            self.categorical_columns = self.pre_encoder.categorical_columns
            self.numerical_columns = self.pre_encoder.numerical_columns
            self.binary_columns = self.pre_encoder.binary_columns

            print("Extended column mappings in TransTab via table_encoder.update().")

        if "num_class" in config:
            self._adapt_to_new_num_class(config["num_class"])

    def _adapt_to_new_num_class(self, num_class: int) -> None:
        r"""Rebuild ``self.clf`` and ``self.loss_fn`` for a new ``num_class``."""
        if not hasattr(self, "clf") or num_class == getattr(self, "num_class", None):
            return
        self.num_class = num_class
        hidden_dim = self.cls_token.hidden_dim
        out_dim = 1 if num_class <= 2 else num_class
        self.clf = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, out_dim),
        )
        if num_class > 2:
            self.loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        else:
            self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
        print(f"Rebuilt classifier for num_class={num_class}.")


class TransTabClassifier(TransTab):
    r"""TransTab with a LayerNorm + Linear classification head on the ``[CLS]`` embedding.

    Supports binary (``num_class <= 2``, BCEWithLogitsLoss) and multi-class
    (CrossEntropyLoss) settings.

    Args:
        categorical_columns (List[str], optional): Categorical column names. Default: ``None``.
        numerical_columns (List[str], optional): Numerical column names. Default: ``None``.
        binary_columns (List[str], optional): Binary column names. Default: ``None``.
        num_class (int): Number of target classes. Default: ``2``.
        hidden_dim (int): Shared embedding dimensionality. Default: ``128``.
        num_layer (int): Number of Transformer layers. Default: ``2``.
        num_attention_head (int): Number of attention heads. Default: ``8``.
        hidden_dropout_prob (float): Dropout probability. Default: ``0.1``.
        ffn_dim (int): Feedforward inner dimension. Default: ``256``.
        activation (str): Feedforward activation. Default: ``"relu"``.
        tokenizer: Pre-trained tokenizer. Default: ``None``.
        **kwargs: Forwarded to :class:`TransTab`.

    Examples::

        >>> from rllm.nn.models import TransTabClassifier
        >>> model = TransTabClassifier(num_class=2, hidden_dim=32)
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
        activation: str = "relu",
        tokenizer=None,
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
            tokenizer=tokenizer,
            **kwargs,
        )

        self.num_class = num_class
        out_dim = 1 if num_class <= 2 else num_class
        self.clf = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, out_dim),
        )

        if num_class > 2:
            self.loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        else:
            self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        x: Union[pd.DataFrame, TableData, Dict[ColType, torch.Tensor]],
        y: Optional[Union[pd.Series, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Args:
            x (TableData): Input table batch.
            y (Tensor or Series, optional): Labels; computes mean loss when provided. Default: ``None``.

        Returns:
            Tuple[Tensor, Optional[Tensor]]: ``(logits [N,1] or [N,K], loss or None)``.
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
    r"""TransTab pre-training via vertical-partition contrastive learning (VPCL).

    Splits each table batch into ``num_partition`` overlapping column subsets,
    encodes each view independently, projects the ``[CLS]`` outputs, and
    optimises a supervised or self-supervised contrastive loss.

    Args:
        categorical_columns (List[str], optional): Categorical column names. Default: ``None``.
        numerical_columns (List[str], optional): Numerical column names. Default: ``None``.
        binary_columns (List[str], optional): Binary column names. Default: ``None``.
        hidden_dim (int): Shared embedding dimensionality. Default: ``128``.
        num_layer (int): Number of Transformer layers. Default: ``2``.
        num_attention_head (int): Number of attention heads. Default: ``8``.
        hidden_dropout_prob (float): Dropout probability. Default: ``0``.
        ffn_dim (int): Feedforward inner dimension. Default: ``256``.
        projection_dim (int): Projection head output dimension. Default: ``128``.
        overlap_ratio (float): Column overlap fraction between partitions, in :math:`[0,1)`. Default: ``0.1``.
        num_partition (int): Number of column subsets per sample. Default: ``2``.
        supervised (bool): Use supervised contrastive loss when ``True``. Default: ``True``.
        temperature (float): Contrastive loss temperature. Default: ``10``.
        base_temperature (float): Base temperature for loss normalisation. Default: ``10``.
        activation (str): Feedforward activation. Default: ``"relu"``.
        tokenizer: Pre-trained tokenizer. Default: ``None``.
        **kwargs: Forwarded to :class:`TransTab`.

    Examples::

        >>> from rllm.nn.models import TransTabForCL
        >>> model = TransTabForCL(hidden_dim=32, num_partition=2)
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
        activation="relu",
        tokenizer=None,
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
            tokenizer=tokenizer,
            **kwargs,
        )
        assert (
            num_partition > 0
        ), f"number of contrastive subsets must be greater than 0, got {num_partition}"
        assert isinstance(
            num_partition, int
        ), f"number of constrative subsets must be int, got {type(num_partition)}"
        assert (
            overlap_ratio >= 0 and overlap_ratio < 1
        ), f"overlap_ratio must be in [0, 1), got {overlap_ratio}"
        self.projection_head = torch.nn.Linear(hidden_dim, projection_dim, bias=False)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.num_partition = num_partition
        self.overlap_ratio = overlap_ratio
        self.supervised = supervised
        self.self_sup_criterion = SelfSupervisedVPCL(
            temperature=self.temperature,
            base_temperature=self.base_temperature,
            similarity="dot",  # "cosine"
        )

        self.sup_criterion = SupervisedVPCL(
            temperature=self.temperature,
            base_temperature=self.base_temperature,
            similarity="dot",
        )

    def forward(self, x, y=None):
        r"""
        Args:
            x (pd.DataFrame or TableData): Input table batch.
            y (Tensor, optional): Labels for supervised contrastive loss. Default: ``None``.

        Returns:
            Tuple[None, Tensor]: ``(None, loss)`` — no logits during pre-training.
        """
        # Extract DataFrame from input
        if isinstance(x, pd.DataFrame):
            df = x
        elif hasattr(x, "df"):  # TableData
            df = x.df
        else:
            raise ValueError(
                f"Expected input type pd.DataFrame or TableData, got {type(x)}"
            )

        # Build positive pairs by splitting columns at DataFrame level
        sub_df_list = self._build_positive_pairs(df, self.num_partition)

        tokenizer_config = TokenizerConfig(
            tokenizer=self.pre_encoder.tokenizer,
            pad_token_id=self.pre_encoder.tokenizer.pad_token_id,
            tokenize_combine=True,
            include_colname=True,
            save_colname_token_ids=True,
        )

        # Process each partition through the full pipeline
        feat_x_list = []
        for sub_df in sub_df_list:
            # Build col_types for this subset
            if hasattr(x, "col_types"):
                sub_col_types = {
                    col: x.col_types[col]
                    for col in sub_df.columns
                    if col in x.col_types
                }
            else:
                sub_col_types = self._infer_col_types(sub_df.columns)

            sub_table = TableData(
                df=sub_df,
                col_types=sub_col_types,
                convert_text_coltypes={ColType.CATEGORICAL},
                tokenizer_config=tokenizer_config,
            )
            # Process through table_encoder
            proc = self.pre_encoder(sub_table)
            emb = proc["embedding"]
            mask = proc["attention_mask"]
            # Add CLS token
            cls_out = self.cls_token(emb, attention_mask=mask)
            emb = cls_out["embedding"]
            mask = cls_out["attention_mask"]

            for conv in self.convs:
                enc_out = conv(x=emb, src_key_padding_mask=mask)
                emb = enc_out

            # Extract CLS representation and project
            feat = enc_out[:, 0, :]  # [bs, hidden_dim]
            feat_proj = self.projection_head(feat)  # [bs, proj_dim]
            feat_x_list.append(feat_proj)

        # Stack multi-view features: [bs, num_partition, proj_dim]
        feat_x_multiview = torch.stack(feat_x_list, dim=1)
        # Compute contrastive loss
        if y is not None and self.supervised:
            labels = y.to(self.device).long()
            loss = self.sup_criterion(feat_x_multiview, labels)
        else:
            loss = self.self_sup_criterion(feat_x_multiview)

        return None, loss

    def _infer_col_types(self, columns):
        col_types = {}
        for col in columns:
            if col in self.pre_encoder.categorical_columns:
                col_types[col] = ColType.TEXT
            elif col in self.pre_encoder.numerical_columns:
                col_types[col] = ColType.NUMERICAL
            elif col in self.pre_encoder.binary_columns:
                col_types[col] = ColType.BINARY
            else:
                col_types[col] = ColType.TEXT
        return col_types

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