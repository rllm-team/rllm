import rllm.transforms.graph_transforms as GT


class RECTTransform(GT.GraphTransform):
    r"""The RECTTransform class is based on the method described in the
    `"Network Embedding with Completely-imbalanced Labels"
    <https://arxiv.org/abs/2007.03545>`__ paper. This transform applies a
    series of transformations to a graph, including feature normalization,
    reduce the dimensionality of the features and adjacency matrix
    normalization.

    Args:
        normalize_features (str): The method for normalizing features
            (default: "l1").
        svd_out_dim (int): The output dimensionality after SVD feature
            reduction (default: 200).
        use_gdc (bool): Whether to use Graph Diffusion Convolution (GDC)
            instead of GCN normalization (default: False).

    Shape:
        - Input: Graph-like data containing ``x`` and ``adj``.
        - Output: Same data type with transformed features and adjacency.

    Examples:
        >>> transform = RECTTransform(svd_out_dim=200, use_gdc=False)
        >>> data = transform(data)
    """

    def __init__(
        self,
        normalize_features: str = "l1",
        svd_out_dim: int = 200,
        use_gdc: bool = False,
    ):
        super().__init__(
            transforms=[
                GT.NormalizeFeatures(normalize_features),
                GT.SVDFeatureReduction(svd_out_dim),
                GT.GDC() if use_gdc else GT.GCNNorm(),
            ]
        )
