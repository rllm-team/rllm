import rllm.transforms.graph_transforms as GT
import rllm.transforms.utils as UT


class GCNTransform(GT.GraphTransform):
    r"""This transform is based on the method described in the
    `"Semi-supervised Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`__ paper.
    GCNTransform applies a series of transformations to a graph,
    including feature normalization and adjacency matrix normalization.

    Args:
        normalize_features (str):
            The method used to normalize the features (default: :obj:`l1`).
    """

    def __init__(self, normalize_features: str = "l1"):
        super().__init__(
            transforms=[
                UT.NormalizeFeatures(normalize_features),
                GT.GCNNorm(),
            ]
        )
