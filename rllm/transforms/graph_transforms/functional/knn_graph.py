from typing import Dict, Optional

import sklearn
import sklearn.neighbors
from torch import Tensor

from rllm.utils.sparse import sparse_mx_to_torch_sparse_tensor


def knn_graph(
    x: Tensor,
    num_neighbors: Optional[int] = 6,
    mode: Optional[str] = "connectivity",
    metric: Optional[str] = "minkowski",
    p: Optional[int] = 2,
    metric_params: Optional[Dict] = None,
    include_self: Optional[bool] = False,
    n_jobs: int = 1,
):
    r"""Creates a k-NN graph based on node features
    Args:
        x (Tensor): The node features.
        num_neighbors (int, optional): Number of neighbors. (default: 6)
        mode (str[`connectivity`, `distance`], optional):
            Type of returned matrix:
            `connectivity` will return the connectivity matrix
            with ones and zeros, while `distance` will return
            the distances between neighbors according to the given metric.
            (default: `connectivity`)
        metric (str[`minkowski`, `cosine`, `l1`, `l2`, ...], optional):
            Metric to use for distance computation.
            Default is `minkowski`, which results in the standard
            Euclidean distance when p = 2.
            (default: `minkowski`)
        p (float): Power parameter for the Minkowski metric (default: `2`).
        metric_paramsdict (dict, optinal):
            Additional keyword arguments
            for the metric function.
            (default: None)
        include_self (bool, optinal): If set to True, the graph will
            contain self-loops. (default: False)
        n_jobs (int): Number of workers to use for computation. (default: 1)
    """

    adj = sklearn.neighbors.kneighbors_graph(
        X=x,
        n_neighbors=num_neighbors,
        mode=mode,
        metric=metric,
        p=p,
        metric_params=metric_params,
        include_self=include_self,
        n_jobs=n_jobs,
    )
    adj_sp = sparse_mx_to_torch_sparse_tensor(adj)
    return adj_sp
