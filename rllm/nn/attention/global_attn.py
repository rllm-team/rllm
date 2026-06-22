import math

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F


class VectorQuantizerEMA(nn.Module):
    r"""Vector quantizer with an exponential-moving-average (EMA) codebook.

    Adapted from the GOAT implementation
    (`<https://github.com/devnkong/GOAT>`__) and used by the global attention
    branch of MetaRTL. Maintains a codebook of centroids that is updated by EMA
    during training, providing a compact set of global keys/values.

    Args:
        num_embeddings (int): Number of codebook entries (centroids).
        embedding_dim (int): Dimensionality of each centroid.
        decay (float): EMA decay rate. (default: :obj:`0.99`)
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, decay: float = 0.99):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._decay = decay

        self.register_buffer(
            "_embedding", torch.randn(num_embeddings, embedding_dim)
        )
        self.register_buffer(
            "_embedding_output", torch.randn(num_embeddings, embedding_dim)
        )
        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("_ema_w", torch.randn(num_embeddings, embedding_dim))

        self.bn = torch.nn.BatchNorm1d(embedding_dim, affine=False)

    def reset_parameters(self):
        r"""Resets all buffers and the internal batch norm statistics."""
        nn.init.normal_(self._embedding, mean=0.0, std=1.0)
        nn.init.normal_(self._embedding_output, mean=0.0, std=1.0)
        nn.init.zeros_(self._ema_cluster_size)
        nn.init.normal_(self._ema_w, mean=0.0, std=1.0)
        self.bn.reset_parameters()

    def get_k(self) -> Tensor:
        r"""Returns the codebook key tensor."""
        return self._embedding_output

    def get_v(self) -> Tensor:
        r"""Returns the codebook value tensor."""
        return self._embedding_output[:, : self._embedding_dim]

    def update(self, x: Tensor) -> Tensor:
        r"""Assign inputs to centroids and EMA-update the codebook.

        Args:
            x (Tensor): Input features of shape :obj:`[N, embedding_dim]`.

        Returns:
            Tensor: Encoding indices of shape :obj:`[N, 1]`.
        """
        inputs_normalized = self.bn(x)
        embedding_normalized = self._embedding

        distances = (
            torch.sum(inputs_normalized**2, dim=1, keepdim=True)
            + torch.sum(embedding_normalized**2, dim=1)
            - 2 * torch.matmul(inputs_normalized, embedding_normalized.t())
        )

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=x.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        if self.training:
            self._ema_cluster_size.data = self._ema_cluster_size * self._decay + (
                1 - self._decay
            ) * torch.sum(encodings, 0)

            dw = torch.matmul(encodings.t(), inputs_normalized)
            self._ema_w.data = self._ema_w * self._decay + (1 - self._decay) * dw

            # Only refresh centroids that have actually received assignments.
            cluster_size = self._ema_cluster_size.clamp(min=1e-5)
            new_embedding = self._ema_w / cluster_size.unsqueeze(1)
            assigned = (self._ema_cluster_size > 1e-3).unsqueeze(1)
            self._embedding.data = torch.where(
                assigned, new_embedding, self._embedding
            )

            running_std = torch.sqrt(self.bn.running_var + 1e-5).unsqueeze(dim=0)
            running_mean = self.bn.running_mean.unsqueeze(dim=0)
            self._embedding_output.data = self._embedding * running_std + running_mean

        return encoding_indices


class GlobalAttn(nn.Module):
    r"""Codebook-based global attention from MetaRTL's fusion stage.

    Each node attends over a learned codebook of centroids (maintained by a
    :class:`VectorQuantizerEMA`) rather than over all other nodes, giving an
    :math:`O(N)` approximation of global self-attention. Attention logits are
    biased by the log of per-centroid assignment counts.

    Args:
        in_dim (int): Input feature dimensionality.
        out_dim (int): Output feature dimensionality.
        num_nodes (int): Total number of nodes (size of the centroid index
            buffer).
        global_dim (int): Dimensionality of the centroid space.
        num_centroids (int): Number of codebook centroids.
        heads (int): Number of attention heads. (default: :obj:`1`)
        attn_dropout (float): Dropout on the attention weights.
            (default: :obj:`0.0`)

    Example:
        >>> import torch
        >>> from rllm.nn.conv.attention import GlobalAttn
        >>> attn = GlobalAttn(32, 32, num_nodes=100, global_dim=16, num_centroids=64)
        >>> x = torch.randn(8, 32)
        >>> idx = torch.arange(8)
        >>> attn(x, idx).shape
        torch.Size([8, 32])
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_nodes: int,
        global_dim: int,
        num_centroids: int,
        heads: int = 1,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.attn_dropout = attn_dropout
        self.num_centroids = num_centroids

        self.vq = VectorQuantizerEMA(num_centroids, global_dim, decay=0.99)
        c = torch.randint(0, num_centroids, (num_nodes,), dtype=torch.long)
        self.register_buffer("c_idx", c)
        self.attn_fn = F.softmax

        if out_dim % heads != 0:
            raise ValueError(f"out_dim ({out_dim}) must be divisible by heads ({heads})")
        attn_channels = out_dim // heads
        self.lin_proj_g = torch.nn.Linear(in_dim, global_dim)
        self.lin_key_g = torch.nn.Linear(global_dim, heads * attn_channels)
        self.lin_query_g = torch.nn.Linear(global_dim, heads * attn_channels)
        self.lin_value_g = torch.nn.Linear(global_dim, heads * attn_channels)
        self.layer_norm_global = nn.LayerNorm(out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.lin_proj_g.reset_parameters()
        self.lin_key_g.reset_parameters()
        self.lin_query_g.reset_parameters()
        self.lin_value_g.reset_parameters()
        self.layer_norm_global.reset_parameters()
        self.vq.reset_parameters()

    @staticmethod
    def _split_heads(t: Tensor, heads: int) -> Tensor:
        # "n (h d) -> h n d"
        n, hd = t.shape
        d = hd // heads
        return t.view(n, heads, d).permute(1, 0, 2)

    @staticmethod
    def _merge_heads(t: Tensor) -> Tensor:
        # "h n d -> n (h d)"
        h, n, d = t.shape
        return t.permute(1, 0, 2).reshape(n, h * d)

    def forward(self, x: Tensor, node_indices: Tensor) -> Tensor:
        r"""Apply codebook-based global attention.

        Args:
            x (Tensor): Node features of shape :obj:`[B, in_dim]`.
            node_indices (Tensor): Global node indices of shape :obj:`[B]`,
                used to update the per-node centroid assignment buffer.

        Returns:
            Tensor: Output of shape :obj:`[B, out_dim]`.
        """
        out = self._global_forward(x, node_indices)
        out = self.layer_norm_global(out)
        return out

    def _global_forward(self, x: Tensor, batch_idx: Tensor) -> Tensor:
        d, h = self.out_dim // self.heads, self.heads
        scale = 1.0 / math.sqrt(d)

        q_x = self.lin_proj_g(x)

        k_x = self.vq.get_k().detach().clone()
        v_x = self.vq.get_v().detach().clone()

        q = self.lin_query_g(q_x)
        k = self.lin_key_g(k_x)
        v = self.lin_value_g(v_x)

        q = self._split_heads(q, h)
        k = self._split_heads(k, h)
        v = self._split_heads(v, h)
        dots = torch.einsum("h i d, h j d -> h i j", q, k) * scale

        c, c_count = self.c_idx.unique(return_counts=True)
        centroid_count = torch.zeros(
            self.num_centroids, dtype=dots.dtype, device=x.device
        )
        centroid_count[c.to(torch.long)] = c_count.to(dots.dtype)
        dots = dots + torch.log(centroid_count.view(1, 1, -1) + 1e-12)
        attn = self.attn_fn(dots, dim=-1)
        attn = F.dropout(attn, p=self.attn_dropout, training=self.training)

        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = self._merge_heads(out)

        if self.training:
            x_idx = self.vq.update(q_x)
            self.c_idx[batch_idx] = x_idx.squeeze().to(torch.long)

        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_dim}, {self.out_dim}, "
            f"heads={self.heads})"
        )
