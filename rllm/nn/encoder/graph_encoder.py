from typing import Any, Dict, List, Optional, Sequence, Type, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from rllm.nn.conv.graph_conv import GCNConv
from .base_encoder import BaseEncoder


class GraphEncoder(BaseEncoder):
    r"""GraphEncoder is a graph-level encoder used by BRIDGE.

    It applies stacked graph convolution layers with dropout and configurable
    activation on all intermediate layers on full-batch adjacency inputs.

    Args:
        in_dim (int): Input dimensionality of the data.
        out_dim (int): Output dimensionality for the encoded data.
        dropout (float): Dropout probability.
        num_layers (int): The number of graph convolution layers.
        graph_conv (Type[torch.nn.Module], optional):
            Convolution module used for graph encoding.
            (default: :obj:`rllm.nn.conv.graph_conv.GCNConv`).
        norm (bool): Whether to enable normalization in graph convolution.
            (default: :obj:`False`).
        norm_layer (str, optional): Feature normalization after each graph
            convolution layer. Supported values are :obj:`"layernorm"`,
            :obj:`"batchnorm1d"`, and :obj:`"none"`.
            If set to :obj:`None` or :obj:`"none"`, no feature
            normalization is applied.
        activation (str): Activation function used on intermediate layers.
            Supported values: :obj:`"relu"`, :obj:`"gelu"`, :obj:`"elu"`,
            :obj:`"leaky_relu"`, :obj:`"selu"`, :obj:`"tanh"`,
            :obj:`"sigmoid"`, :obj:`"none"`.
            (default: :obj:`"relu"`).
        pre_encoder (Type[torch.nn.Module], optional):
            Optional pre-encoder to process node features before graph
            convolutions.
            (default: :obj:`None`).
        pre_encoder_kwargs (Dict[str, Any], optional):
            Extra keyword args passed to pre-encoder constructor.
        pre_encoder_out_dim (int, optional): Optional output dimension of
            pre-encoder. If set, it is used as the first graph-conv input
            dimension.

    Returns:
        This class does not return tensors in ``__init__``.
        The ``forward`` method returns graph node embeddings.
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.5,
        num_layers: int = 2,
        graph_conv: Type[torch.nn.Module] = GCNConv,
        norm: bool = False,
        norm_layer: Optional[str] = "none",
        activation: str = "relu",
        pre_encoder: Optional[Type[torch.nn.Module]] = None,
        pre_encoder_kwargs: Optional[Dict[str, Any]] = None,
        pre_encoder_out_dim: Optional[int] = None,
        last_layer_activation: bool = False,
    ) -> None:
        super().__init__(num_layers=num_layers)
        self.dropout = dropout
        self.activation = self.get_activation(activation)
        self.last_layer_activation = last_layer_activation

        dims = self.build_layer_dims(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            hidden_default_dim=in_dim,
            num_layers=num_layers,
        )

        # Initialize pre-encoder if provided
        pre_encoder_kwargs = dict(pre_encoder_kwargs or {})
        if pre_encoder is not None:
            if pre_encoder_out_dim is not None:
                pre_encoder_kwargs["out_dim"] = pre_encoder_out_dim
            self.pre_encoder = pre_encoder(in_dim=in_dim, **pre_encoder_kwargs)
        else:
            self.pre_encoder = None

        if pre_encoder_out_dim is not None:
            dims[0] = pre_encoder_out_dim

        norm_cls = self.resolve_norm_layer(norm_layer)

        self.norms = nn.ModuleList()
        for layer_in_dim, layer_out_dim in zip(dims[:-1], dims[1:]):
            self.convs.append(
                graph_conv(in_dim=layer_in_dim, out_dim=layer_out_dim, normalize=norm)
            )
            self.norms.append(norm_cls(layer_out_dim))

    def _prepare_layer_adjs(
        self,
        adj: Union[Tensor, Sequence[Tensor]],
    ) -> List[Tensor]:
        if isinstance(adj, Tensor):
            return [adj] * self.num_layers

        layer_adjs = list(adj)
        if len(layer_adjs) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} sampled adjs for {self.num_layers} "
                f"layers, but got {len(layer_adjs)}."
            )
        return list(reversed(layer_adjs))

    def _apply_layer(
        self,
        x: Tensor,
        conv: nn.Module,
        norm: nn.Module,
        layer_adj: Tensor,
        apply_activation: bool,
    ) -> Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = conv(x, layer_adj)
        x = norm(x)
        if apply_activation:
            x = self.activation(x)
        return x

    def forward(
        self,
        x: Tensor,
        adj: Union[Tensor, Sequence[Tensor]],
    ) -> Tensor:
        """Apply stacked graph convolutions to node features.

        Args:
            x (Tensor): Node features of shape [num_nodes, in_dim].
                        adj (Union[Tensor, Sequence[Tensor]]):
                - Full-batch adjacency matrix of shape
                  [num_nodes, num_nodes].
                - Layer-wise sampled adjacency list returned by neighbor
                  samplers. The list order follows sampler hops and is
                  consumed in reverse order during message passing.

        Returns:
            Tensor: Output node embeddings of shape [num_nodes, out_dim].
        """
        if self.pre_encoder is not None:
            x = self.pre_encoder(x)
            expected_in_dim = getattr(self.convs[0], "in_dim", None)
            if expected_in_dim is not None:
                self.validate_feature_last_dim(
                    x=x,
                    expected_dim=expected_in_dim,
                    allow_dict=False,
                )

        layer_adjs = self._prepare_layer_adjs(adj)
        last_idx = len(self.convs) - 1
        for idx, (conv, norm, layer_adj) in enumerate(
            zip(self.convs, self.norms, layer_adjs)
        ):
            apply_activation = idx < last_idx or self.last_layer_activation
            x = self._apply_layer(
                x=x,
                conv=conv,
                norm=norm,
                layer_adj=layer_adj,
                apply_activation=apply_activation,
            )
        return x
