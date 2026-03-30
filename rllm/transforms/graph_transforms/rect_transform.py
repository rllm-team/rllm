import rllm.transforms.graph_transforms as GT


class RECTTransform(GT.GraphTransform):
    r"""The RECTTransform class is based on the method described in the
    `"Network Embedding with Completely-imbalanced Labels"
    <https://arxiv.org/abs/2007.03545>`__ paper. This transform applies a
    series of transformations to a graph, including feature normalization,
    reduce the dimensionality of the features and adjacency matrix
    normalization.

    Args:
        normalize_features (str): Method for feature normalization.
            (default: :obj:`"l1"`)
        svd_out_dim (int): The output dimensionality after SVD feature
            reduction. (default: :obj:`200`)
        use_gdc (bool): Whether to use Graph Diffusion Convolution (GDC)
            instead of GCN normalization. (default: :obj:`False`)
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
