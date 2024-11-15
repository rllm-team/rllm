from typing import Optional, Union

from rllm.types import ColType
from rllm.data.graph_data import GraphData
from rllm.data.table_data import TableData
from rllm.transforms.graph_transforms import BaseTransform
from rllm.transforms.graph_transforms.functional import knn_graph


class KNNGraph(BaseTransform):  # TODO: add force_undirected option.
    r"""Creates a k-NN graph based on node features

    Args:
        k (int, optional): The number of neighbors. (default: 6)
        mode (str[`connectivity`, `distance`], optional):
            Type of returned matrix: `connectivity` will return the
            connectivity matrix with ones and zeros, while `distance`
            will return the distances between neighbors
            according to the given metric.
            (default: `connectivity`)
        metric (str[`minkowski`, `cosine`, `l1`, `l2`, ...], optional):
            Metric to use for distance computation.
            Default is `minkowski`, which results in the
            standard Euclidean distance when p = 2.
            (default: `minkowski`)
        metric_paramsdict (dict, optinal):
            Additional keyword arguments
            for the metric function.
            (default: None)
        include_self (bool, optinal):
            If set to True, the graph will contain self-loops. (default: False)
        n_jobs (int): Number of workers to use for computation. (default: 1)
    """

    def __init__(
        self,
        k: Optional[int] = 6,
        mode: Optional[str] = "connectivity",
        metric: Optional[str] = "minkowski",
        p: Optional[int] = 2,
        metric_params: Optional[dict] = None,
        include_self: Optional[bool] = False,
        n_jobs: int = 1,
    ):
        self.k = k
        self.mode = mode
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.include_self = include_self
        self.n_jobs = n_jobs

    def forward(self, data: Union[GraphData, TableData]) -> GraphData:
        if isinstance(data, GraphData):
            assert data.x is not None
            x = data.x
        elif isinstance(data, TableData):
            x = data[ColType.CATEGORICAL]

        knn_adj = knn_graph(
            x,
            self.k,
            self.mode,
            self.metric,
            self.p,
            self.metric_params,
            self.include_self,
            self.n_jobs,
        )

        if isinstance(data, GraphData):
            data.adj = knn_adj
        else:
            data = GraphData(x=x, adj=knn_adj)
        return data
