import torch
from torch import Tensor
import torch.nn.functional as F

from rllm.nn.conv.graph_conv import GCNConv


class RECT_L(torch.nn.Module):
    r"""The RECT model, or more specifically its supervised part RECT-L,
    from the `"Network Embedding with Completely-imbalanced Labels"
    <https://arxiv.org/abs/2007.03545>`__ paper.
    In particular, a GCN model is trained that reconstructs semantic class
    knowledge.

    Args:
        in_dim (int): Size of each input sample.
        hidden_dim (int): Intermediate size of each sample.
        dropout (float, optional): The dropout probability.
            (default: :obj:`0.0`)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.prelu = torch.nn.PReLU()
        self.conv = GCNConv(in_dim, hidden_dim)
        self.lin = torch.nn.Linear(hidden_dim, in_dim)
        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.conv.reset_parameters()
        self.lin.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin.weight.data)

    def forward(self, x: Tensor, adj: Tensor):
        x = self.prelu(self.conv(x, adj))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x)

    @torch.jit.export
    def embed(self, x: Tensor, adj: Tensor):
        with torch.no_grad():
            return self.prelu(self.conv(x, adj))

    @torch.jit.export
    def get_semantic_labels(self, x: Tensor, y: Tensor, mask: Tensor):
        r"""Replaces the original labels by their class-centers."""
        device = x.device
        with torch.no_grad():
            x = x[mask]
            y = y[mask]
            classes = y.unique()
            class_centers = torch.zeros(classes.max() + 1, x.shape[1]).to(device)
            for ci in classes:
                class_centers[ci] = x[y == ci].mean(0)
            return class_centers[y]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_dim}, " f"{self.hidden_dim})"
