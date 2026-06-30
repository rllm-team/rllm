import torch
from torch import Tensor
from torch import nn
from typing import Callable

from rllm.nn.models.metartl.global_attn import GlobalAttn
from rllm.nn.models.metartl.linear_per_metapath import LinearPerMetapath
from rllm.nn.models.metartl.semantic_transformer import SemanticTransformer


class MetaPathFusion(nn.Module):
    r"""Meta-path fusion model (stage two) of MetaRTL
    `"MetaRTL: Meta-path Attention Enhanced Relational Table Learning"`.

    Consumes the per-node meta-path feature tensor of shape :obj:`[B, M, D]`
    produced by
    :class:`~rllm.nn.models.metartl.MetaPathProp`, and fuses the
    meta-paths through two complementary branches:

    - a **local** branch using a :class:`SemanticTransformer` to mix
      information across meta-paths, and
    - a **global** branch using a codebook-based :class:`GlobalAttn`.

    The two branch outputs are concatenated, projected, and passed through an
    MLP head to produce predictions.

    Args:
        num_nodes (int): Total number of (training) target nodes, used to size
            the global attention centroid buffer.
        in_dim (int): Input meta-path feature dimensionality :obj:`D`.
        hidden_dim (int): Hidden dimensionality. Should equal :obj:`in_dim`
            because of the global-branch residual connection.
        num_metapaths (int): Number of meta-paths :obj:`M`.
        out_dim (int): Output dimensionality.
        num_centroids (int): Number of global-attention centroids.
            (default: :obj:`4096`)
        num_heads (int): Number of attention heads. (default: :obj:`4`)
        att_drop (float): Attention dropout. (default: :obj:`0.3`)
        readout_drop (float): Dropout in the readout head. (default: :obj:`0.3`)
        readout_layers (int): Number of layers in the readout head.
            (default: :obj:`1`)
        norm (str): Normalization for the readout head. (default: :obj:`"batch_norm"`)
        num_proj_layers (int): Number of per-meta-path projection layers.
            (default: :obj:`2`)
        proj_drop (float): Dropout in the projection layers. (default: :obj:`0.3`)

    Example:
        >>> import torch
        >>> from rllm.nn.models import MetaPathFusion
        >>> model = MetaPathFusion(
        ...     num_nodes=1000, in_dim=128, hidden_dim=128,
        ...     num_metapaths=8, out_dim=1,
        ... )
        >>> x = torch.randn(16, 8, 128)
        >>> idx = torch.arange(16)
        >>> model(x, idx).shape
        torch.Size([16, 1])
    """

    def __init__(
        self,
        num_nodes: int,
        in_dim: int,
        hidden_dim: int,
        num_metapaths: int,
        out_dim: int,
        num_centroids: int = 4096,
        num_heads: int = 4,
        att_drop: float = 0.3,
        readout_drop: float = 0.3,
        readout_layers: int = 1,
        norm: str = "batch_norm",
        num_proj_layers: int = 2,
        proj_drop: float = 0.3,
    ):
        super().__init__()

        proj_layers = [
            LinearPerMetapath(in_dim, hidden_dim, num_metapaths),
            nn.LayerNorm([num_metapaths, hidden_dim]),
            nn.PReLU(),
            nn.Dropout(proj_drop),
        ]
        for _ in range(num_proj_layers - 1):
            proj_layers += [
                LinearPerMetapath(hidden_dim, hidden_dim, num_metapaths),
                nn.LayerNorm([num_metapaths, hidden_dim]),
                nn.PReLU(),
                nn.Dropout(proj_drop),
            ]
        self.feature_projection = nn.Sequential(*proj_layers)

        # local branch
        self.local_attn = SemanticTransformer(
            n_channels=hidden_dim,
            num_heads=num_heads,
            att_drop=att_drop,
        )
        self.local_fc = nn.Linear(hidden_dim * num_metapaths, hidden_dim)

        # global branch
        self.global_attn = GlobalAttn(
            in_dim=hidden_dim,
            out_dim=hidden_dim,
            num_nodes=num_nodes,
            global_dim=hidden_dim // 2,
            num_centroids=num_centroids,
            heads=num_heads,
            attn_dropout=att_drop,
        )

        self.ff = nn.Linear(hidden_dim * 2, hidden_dim)
        self.head = self._build_mlp_head(
            in_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=readout_layers,
            norm=norm,
            dropout=readout_drop,
        )

        self.reset_parameters()

    def _build_mlp_head(
        self,
        in_dim: int,
        out_dim: int,
        num_layers: int,
        norm: str,
        dropout: float,
    ) -> nn.Sequential:
        if num_layers <= 1:
            return nn.Sequential(nn.Linear(in_dim, out_dim))

        hidden_dim = in_dim // 2
        norm_layer: Callable[[int], nn.Module] = (
            nn.BatchNorm1d if norm == "batch_norm" else nn.LayerNorm
        )

        layers = [
            nn.Linear(in_dim, hidden_dim),
            norm_layer(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        for _ in range(num_layers - 2):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                norm_layer(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        layers.append(nn.Linear(hidden_dim, out_dim))
        return nn.Sequential(*layers)

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for layer in self.feature_projection:
            reset_fn = getattr(layer, "reset_parameters", None)
            if callable(reset_fn):
                reset_fn()
        self.local_attn.reset_parameters()
        self.local_fc.reset_parameters()
        self.global_attn.reset_parameters()
        self.ff.reset_parameters()
        for layer in self.head:
            reset_fn = getattr(layer, "reset_parameters", None)
            if callable(reset_fn):
                reset_fn()

    def forward(self, x: Tensor, batch_idx: Tensor) -> Tensor:
        r"""Fuse meta-path features and predict.

        Args:
            x (Tensor): Meta-path features of shape :obj:`[B, M, D]`.
            batch_idx (Tensor): Global node indices of shape :obj:`[B]` for the
                global attention branch.

        Returns:
            Tensor: Predictions of shape :obj:`[B, out_dim]`.
        """
        res = x[:, 0, :]
        x = self.feature_projection(x)  # [B, M, C']

        # local
        local_x = self.local_attn(x)
        local_x = local_x.reshape(local_x.size(0), -1)
        local_x = self.local_fc(local_x)  # [B, C']

        # global
        global_x = x[:, 0, :]  # [B, C']
        global_x = self.global_attn(global_x, node_indices=batch_idx)
        global_x = global_x + res  # [B, C']

        out = torch.cat([local_x, global_x], dim=1)
        out = self.ff(out)
        out = self.head(out)
        return out
