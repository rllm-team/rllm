import rllm.transforms.graph_transforms as GT


class GCNTransform(GT.GraphTransform):
    r"""Preprocessing pipeline used by the original GCN model.

    This transform is based on
    `"Semi-supervised Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`__ paper.
    GCNTransform applies a series of transformations to a graph,
    including:

    1. Feature Normalization
    2. Adjacency Matrix Normalization
        a. Adding Self-Loops
        b. Symmetric Normalization

    Args:
        normalize_features (str): Feature normalization method passed to
            :class:`NormalizeFeatures`. (default: :obj:`"l1"`)
    """

    def __init__(self, normalize_features: str = "l1"):
        super().__init__(
            transforms=[
                GT.NormalizeFeatures(normalize_features),
                GT.GCNNorm(),
            ]
        )
