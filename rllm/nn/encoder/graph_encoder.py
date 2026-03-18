from typing import Any, Dict, List, Optional, Type, Union

import torch
from torch import Tensor
import torch.nn.functional as F

from rllm.nn.conv.graph_conv import GCNConv
from .base_encoder import BaseEncoder


class GraphEncoder(BaseEncoder):
    r"""GraphEncoder is a graph-level encoder used by BRIDGE.

    It applies stacked graph convolution layers with dropout and configurable
    activation on all
    intermediate layers, supporting both full-batch and mini-batch adjacency
    inputs.

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
        activation (str): Activation function used on intermediate layers.
            Supported values: :obj:`"relu"`, :obj:`"gelu"`, :obj:`"elu"`,
            :obj:`"leaky_relu"`, :obj:`"selu"`, :obj:`"tanh"`,
            :obj:`"sigmoid"`, :obj:`"identity"`, :obj:`"none"`.
            (default: :obj:`"relu"`).
        pre_encoder (Type[torch.nn.Module], optional):
            Optional pre-encoder to process node features before graph
            convolutions. If provided, its output should match :obj:`in_dim`.
            (default: :obj:`None`).
        pre_encoder_kwargs (Dict[str, Any], optional):
            Extra keyword args passed to pre-encoder constructor.

    Returns:
        This class does not return tensors in ``__init__``.
        The ``forward`` method returns graph node embeddings.
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        dropout: float = 0.5,
        num_layers: int = 2,
        graph_conv: Type[torch.nn.Module] = GCNConv,
        norm: bool = False,
        activation: str = "relu",
        pre_encoder: Optional[Type[torch.nn.Module]] = None,
        pre_encoder_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(num_layers=num_layers)
        self.dropout = dropout
        self.activation = self.get_activation(activation)

        pre_encoder_kwargs = pre_encoder_kwargs or {}
        if pre_encoder is not None:
            self.pre_encoder_module = pre_encoder(
                in_dim=in_dim,
                out_dim=in_dim,
                **pre_encoder_kwargs,
            )
        else:
            self.pre_encoder_module = None

        for _ in range(num_layers - 1):
            self.convs.append(graph_conv(in_dim=in_dim, out_dim=in_dim, normalize=norm))
        self.convs.append(graph_conv(in_dim=in_dim, out_dim=out_dim, normalize=norm))

    def forward(
        self,
        x: Tensor,
        adj: Union[Tensor, List[Tensor]],
    ) -> Tensor:
        """Apply stacked graph convolutions to node features.

        Args:
            x (Tensor): Node features.
            adj (Union[Tensor, List[Tensor]]): Full-batch adjacency tensor or
                sampled adjacency tensors for mini-batch training.

        Returns:
            Tensor: Output node embeddings.
        """
        if self.pre_encoder_module is not None:
            x = self.pre_encoder_module(x)

        # Full batch training or full test
        if isinstance(adj, Tensor):
            for conv in self.convs[:-1]:
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.activation(conv(x, adj))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[-1](x, adj)
            return x
        # Batch training
        elif isinstance(adj, list):
            for i, conv in enumerate(self.convs[:-1]):
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.activation(conv(x, adj[-i - 1]))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[-1](x, adj[0])
            return x

        raise TypeError(f"Expected adj to be Tensor or List[Tensor], got {type(adj)}")
