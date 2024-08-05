from __future__ import annotations

from typing import Any

from torch import Tensor
from torch.nn import LayerNorm, Linear, Module, ReLU, Sequential

from rllm.types import ColType, StatType
from rllm.nn.conv.ft_transformer_conv import FTTransformerConvs
from rllm.nn.encoder.coltype_encoder import (
    CategoricalTransform,
    NumericalTransform,
    ColTypeTransform,
)
from rllm.nn.encoder.tabletype_encoder import TableTypeTransform


class FTTransformer(Module):
    r"""The FT-Transformer model introduced in the
    `"Revisiting Deep Learning Models for Tabular Data"
    <https://arxiv.org/abs/2106.11959>`_ paper.

    .. note::

        For an example of using FTTransformer, see `examples/revisiting.py
        <https://github.com/pyg-team/pytorch-frame/blob/master/examples/
        revisiting.py>`_.

    Args:
        hidden_dim (int): Hidden channel dimensionality
        output_dim (int): Output channels dimensionality
        layers (int): Number of layers.  (default: :obj:`3`)
        col_stats_dict(dict[:class:`rllm.types.ColType`, list[dict[str, Any]]):
             A dictionary that maps column types into stats.
             Available as :obj:`dataset.stats_dict`.
        stype_transform_dict
            (dict[:class:`rllm.types.ColType`,
            :class:`rllm.nn.encoder.ColTypetransform`], optional):
            A dictionary mapping stypes into their column type transforms.
            (default: :obj:`None`, will call
            :class:`rllm.nn.encoder.Categoricaltransform()` for categorical
            feature and :class:`rllm.nn.encoder.Lineartransform()`
            for numerical feature)
    """
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        layers: int,
        col_stats_dict: dict[str, dict[StatType, Any]],
        col_types_transform_dict: dict[ColType, ColTypeTransform]
        | None = None,
    ) -> None:
        super().__init__()
        if layers <= 0:
            raise ValueError(
                f"layers must be a positive integer (got {layers})")
        # 1. Create column transforms
        if col_types_transform_dict is None:
            col_types_transform_dict = {
                ColType.CATEGORICAL: CategoricalTransform(),
                ColType.NUMERICAL: NumericalTransform(type='linear'),
            }

        self.transform = TableTypeTransform(
            out_channels=hidden_dim,
            col_stats_dict=col_stats_dict,
            col_types_transform_dict=col_types_transform_dict,
        )
        # 2. Prepare table convs backbone
        self.backbone = FTTransformerConvs(dim=hidden_dim,
                                           layers=layers)
        # 3. Prepare decoder
        self.decoder = Sequential(
            LayerNorm(hidden_dim),
            ReLU(),
            Linear(hidden_dim, output_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.transform.reset_parameters()
        self.backbone.reset_parameters()
        for m in self.decoder:
            if not isinstance(m, ReLU):
                m.reset_parameters()

    def forward(self, feat_dict) -> Tensor:
        r"""Transforming feat_dict object into output prediction.

        Args:
            tf (TensorFrame):
                Input: feat_dict(dict[:class:`rllm.types.ColType`, Tensor])

        Returns:
            torch.Tensor: Output of shape [batch_size, output_dim].
        """
        x, _ = self.transform(feat_dict)
        x, x_cls = self.backbone(x)
        out = self.decoder(x_cls)
        return out
