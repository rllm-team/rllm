import torch
from torch import Tensor


def svd_feature_reduction(X: Tensor, out_dim: int):
    r"""Dimensionality reduction of node features via Singular Value
    Decomposition (SVD).

    Args:
        x (Tensor): Node feature matrix.
        out_dim (int): The dimensionlity of node features after
            reduction.
    """
    if X.size(-1) > out_dim:
        U, S, _ = torch.linalg.svd(X)
        X = torch.mm(U[:, :out_dim], torch.diag(S[:out_dim]))
    return X
