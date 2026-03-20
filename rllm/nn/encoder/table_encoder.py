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

    It optionally applies a pre-encoder to each column type, then stacks
    table convolution layers.

    Args:
        in_dim (int): Input dimensionality of the table data.
        out_dim (int): Output dimensionality for the encoded table data.
        hidden_dim (int, optional): Hidden dimensionality for intermediate
            layers. If :obj:`None`, defaults to :obj:`out_dim`.
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
        pre_encoder_out_dim (int, optional): Optional output dimension of the
            pre-encoder. If set, it is used as the first table-conv input
            dimension.
        pre_encoder_return_dict (bool, optional):
            Whether to ask pre-encoder to return a feature dict.
            (default: :obj:`False`).
        norm_layer (str, optional): Feature normalization after each table
            convolution layer. Supported values are :obj:`"layernorm"`,
            :obj:`"batchnorm1d"`, and :obj:`"none"`.
            If set to :obj:`None` or :obj:`"none"`, no feature
            normalization is applied.
        activation (str, optional):
            Activation function to apply after each convolution layer.
            Supported values: :obj:`"relu"`, :obj:`"gelu"`, :obj:`"elu"`,
            :obj:`"leaky_relu"`, :obj:`"selu"`, :obj:`"tanh"`,
            :obj:`"sigmoid"`, :obj:`"none".
            (default: :obj:`"none"`).
        pre_encoder_kwargs (Dict[str, Any], optional):
            Extra keyword args passed to pre-encoder constructor.
        table_conv_kwargs (Dict[str, Any], optional):
            Extra keyword args passed to table convolution constructor.

    Returns:
        This class does not return tensors in ``__init__``.
        The ``forward`` method returns pooled table embeddings.
    """

    def __init__(
        self,
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
        """Encode a table into feature representations.

        Args:
            x (Union[TableData, Dict[ColType, Tensor]]): Input table data object.

        Returns:
            Union[Tensor, Dict[ColType, Tensor]]: Encoded output after table
            convolution layers. The output type matches the pre-encoder
            configuration.
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
