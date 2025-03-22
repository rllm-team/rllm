from torch import linalg as LA
from torch import Tensor


def normalize_features(X: Tensor, norm: str = "l2", return_norm: bool = False):
    r"""Scale input vectors individually to unit norm.
    Args:
        X (Tensor): The input vectors.
        norm (str): The norm to use to normalize each non zero sample,
            *e.g.*, `l1`, `l2`, `sum`. (default: `l2`).
        return_norm (bool): Whether to return the computed norms.
            (default: `False`).
    """
    if X.is_sparse:
        X = X.to_dense()

    if norm == "l1":
        norms = LA.norm(X, ord=1, dim=1, keepdim=True)
    elif norm == "l2":
        norms = LA.norm(X, dim=1, keepdim=True)
    elif norm == "sum":
        X -= X.min()
        norms = X.sum(dim=-1, keepdim=True)

    X = X.div_(norms.clamp_(min=1.0))

    if return_norm:
        norms = norms.squeeze(1)
        return X, norms
    else:
        return X
