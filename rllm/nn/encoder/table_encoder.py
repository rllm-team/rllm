<<<<<<< HEAD
import inspect
from typing import Any, Dict, List, Optional, Type, Union

import torch
from torch import Tensor, nn

from rllm.data import TableData
from rllm.types import ColType
from rllm.nn.conv.table_conv import TabTransformerConv
from .base_encoder import BaseEncoder
from .pre_encoder.tab_transformer_pre_encoder import TabTransformerPreEncoder


class TableEncoder(BaseEncoder):
    r"""TableEncoder is a table-level encoder used by BRIDGE.

    It first applies a pre-encoder to each column type, then stacks table
    convolution layers and finally pools across columns.

    Args:
        in_dim (int): Input dimensionality of the table data.
        out_dim (int): Output dimensionality for the encoded table data.
        num_layers (int, optional): Number of convolution layers.
            (default: :obj:`1`).
        metadata (Dict[ColType, List[Dict[str, Any]]], optional):
            Metadata for each column type.
        table_conv (Type[torch.nn.Module], optional):
            The convolution module used for table encoding.
            (default: :obj:`rllm.nn.conv.table_conv.TabTransformerConv`).
        pre_encoder (Type[torch.nn.Module], optional):
            The pre-encoder used before table convolution.
            (default: :obj:`rllm.nn.encoder.TabTransformerPreEncoder`).
        pre_encoder_return_dict (bool, optional):
            Whether to ask pre-encoder to return a feature dict.
            (default: :obj:`True`).
        activation (str, optional):
            Activation function to apply after each convolution layer.
            Supported values: :obj:`"relu"`, :obj:`"tanh"`, :obj:`"none"`.
            (default: :obj:`"none"`).
        pre_encoder_kwargs (Dict[str, Any], optional):
            Extra keyword args passed to pre-encoder constructor.
        table_conv_kwargs (Dict[str, Any], optional):
            Extra keyword args passed to table convolution constructor.

    Returns:
        This class does not return tensors in ``__init__``.
        The ``forward`` method returns pooled table embeddings.
=======
from __future__ import annotations
from typing import Any, Dict, List, Union
from abc import ABC

import torch
from torch import Tensor

from ..col_encoder import ColEncoder
from rllm.types import ColType


class TableEncoder(torch.nn.Module, ABC):
    r"""The TableEncoder class is designed to transform table data by encoding
    each column type tensor into embeddings and performing the final
    concatenation. It supports different types of column encoders for
    categorical and numerical features, allowing for flexible and
    efficient preprocessing of tabular data.

    Args:
        out_dim (int): Output dimensionality.
        metadata(Dict[ColType, List[Dict[str, Any]]]):Metadata for each column
            type, specifying the statistics and properties of the columns.
        col_encoder_dict
            (Dict[:class:`rllm.types.ColType`,
            :class:`rllm.nn.encoder.ColEncoder]):
            A dictionary that maps :class:`rllm.types.ColType` into
            :class:`rllm.nn.encoder.ColEncoder` class. Only
            parent :class:`stypes <rllm.types.ColType>` are supported
            as keys.

    Returns:
        This class does not return a tensor in ``__init__``.
        The ``forward`` method returns either a concatenated embedding tensor
        or a dictionary of per-column-type embeddings.

    Example:
        >>> from rllm.nn.encoder import PreEncoder
        >>> # Usually instantiated through concrete subclasses.
>>>>>>> main
    """

    def __init__(
        self,
<<<<<<< HEAD
        in_dim: int,
        out_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 1,
        metadata: Dict[ColType, List[Dict[str, Any]]] = None,
        table_conv: Type[torch.nn.Module] = TabTransformerConv,
        pre_encoder: Optional[Type[torch.nn.Module]] = TabTransformerPreEncoder,
        pre_encoder_out_dim: Optional[int] = None,
        pre_encoder_return_dict: bool = False,
        norm_layer: Optional[str] = "none",
        activation: str = "none",
        pre_encoder_kwargs: Optional[Dict[str, Any]] = None,
        table_conv_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(num_layers=num_layers)

        self.pre_encoder_return_dict = pre_encoder_return_dict
        self.activation = self.get_activation(activation)

        dims = self.build_layer_dims(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            hidden_default_dim=out_dim,
            num_layers=num_layers,
        )
        self._first_layer_in_dim = dims[0]

        pre_encoder_kwargs = dict(pre_encoder_kwargs or {})
        table_conv_kwargs = dict(table_conv_kwargs or {})

        if pre_encoder is not None:
            pe_out_dim = dims[0] if pre_encoder_out_dim is None else pre_encoder_out_dim
            self.pre_encoder = pre_encoder(
                out_dim=pe_out_dim,
                metadata=metadata,
                **pre_encoder_kwargs,
            )
            dims[0] = pe_out_dim
            self._first_layer_in_dim = pe_out_dim
        else:
            self.pre_encoder = None

        norm_cls = self.resolve_norm_layer(norm_layer)

        self.input_projs = nn.ModuleList()
        self.norms = nn.ModuleList()

        init_params = inspect.signature(table_conv.__init__).parameters
        supports_in_out = "in_dim" in init_params and "out_dim" in init_params

        for layer_in_dim, layer_out_dim in zip(dims[:-1], dims[1:]):
            if supports_in_out:
                self.input_projs.append(nn.Identity())
                conv = table_conv(
                    in_dim=layer_in_dim,
                    out_dim=layer_out_dim,
                    **table_conv_kwargs,
                )
            else:
                if layer_in_dim == layer_out_dim:
                    self.input_projs.append(nn.Identity())
                else:
                    self.input_projs.append(nn.Linear(layer_in_dim, layer_out_dim))
                conv = table_conv(conv_dim=layer_out_dim, **table_conv_kwargs)

            self.convs.append(conv)
            self.norms.append(norm_cls(layer_out_dim))

    def _apply_activation(
        self,
        x: Union[Tensor, Dict[ColType, Tensor]],
    ) -> Union[Tensor, Dict[ColType, Tensor]]:
        if isinstance(x, dict):
            return {k: self.activation(v) for k, v in x.items()}
        return self.activation(x)

    def _apply_norm(
        self,
        x: Union[Tensor, Dict[ColType, Tensor]],
        norm: nn.Module,
    ) -> Union[Tensor, Dict[ColType, Tensor]]:
        def apply_one(feat: Tensor) -> Tensor:
            if isinstance(norm, nn.BatchNorm1d) and feat.dim() > 2:
                return norm(feat.transpose(1, -1)).transpose(1, -1)
            return norm(feat)

        if isinstance(x, dict):
            return {k: apply_one(v) for k, v in x.items()}
        return apply_one(x)

    def _apply_layer(
        self,
        x: Union[Tensor, Dict[ColType, Tensor]],
        conv: nn.Module,
        norm: nn.Module,
        input_proj: nn.Module,
        apply_activation: bool,
    ) -> Union[Tensor, Dict[ColType, Tensor]]:
        if isinstance(x, dict):
            x_in = {k: input_proj(v) for k, v in x.items()}
        else:
            x_in = input_proj(x)
        h = conv(x_in)
        h = self._apply_norm(h, norm)
        x = h
        if apply_activation:
            x = self._apply_activation(x)
        return x

    def _validate_pre_encoder_output_dim(
        self,
        x: Union[Tensor, Dict[ColType, Tensor]],
    ) -> None:
        self.validate_feature_last_dim(
            x=x,
            expected_dim=self._first_layer_in_dim,
            allow_dict=True,
        )

    def forward(
        self, x: Union[TableData, Dict[ColType, Tensor]]
    ) -> Union[Tensor, Dict[ColType, Tensor]]:
        """Encode a table into a pooled feature representation.

        Args:
            x (Union[TableData, Dict[ColType, Tensor]]): Input table data object.

        Returns:
            Union[Tensor, Dict[ColType, Tensor]]: Encoded output after table
            backbone. If pooling is :obj:`"none"`, this returns the raw
            convolution output without pooling.
        """
        if isinstance(x, TableData):
            x = x.feat_dict

        if self.pre_encoder is not None:
            x = self.pre_encoder(x, return_dict=self.pre_encoder_return_dict)
            self._validate_pre_encoder_output_dim(x)

        last_idx = len(self.convs) - 1
        for idx, conv in enumerate(self.convs):
            x = self._apply_layer(
                x=x,
                conv=conv,
                norm=self.norms[idx],
                input_proj=self.input_projs[idx],
                apply_activation=idx < last_idx,
            )
        return x
=======
        out_dim: int,
        metadata: Dict[ColType, List[Dict[str, Any]]],
        col_encoder_dict: Dict[ColType, ColEncoder],
    ) -> None:
        super().__init__()

        self.metadata = metadata
        self.col_encoder_dict = torch.nn.ModuleDict()

        for col_type, col_encoder in col_encoder_dict.items():
            if col_type not in col_encoder.supported_types:
                raise ValueError(
                    f"{col_encoder} does not " f"support encoding {col_type}."
                )
            # Set attributes
            if col_encoder.out_dim is None:
                col_encoder.out_dim = out_dim
            if col_type in metadata.keys():
                col_encoder.stats_list = metadata[col_type]
                self.col_encoder_dict[col_type.value] = col_encoder
                col_encoder.post_init()
        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters for all encoders in the encoder_dict."""
        for col_encoder in self.col_encoder_dict.values():
            col_encoder.reset_parameters()

    def forward(
        self,
        feat_dict: Dict[ColType, Tensor],
        return_dict: bool = False,
    ) -> Union[Tensor, Dict[ColType, Tensor]]:
        feat_encoded = {}
        for col_type in feat_dict.keys():
            feat = feat_dict[col_type]
            if col_type.value in self.col_encoder_dict.keys():
                x = self.col_encoder_dict[col_type.value](feat)
                feat_encoded[col_type] = x
            else:
                feat_encoded[col_type] = feat

        if return_dict:
            return feat_encoded

        feat_list = list(feat_encoded.values())
        return torch.cat(feat_list, dim=1)
>>>>>>> main
