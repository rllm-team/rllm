import rllm.transforms.utils as UT
import rllm.transforms.graph_transforms as GT


class RECTTransform(GT.GraphTransform):
    def __init__(
        self,
        normalize_features: str = "l1",
        svd_out_dim: int = 200,
        use_gdc=False,
    ):
        super().__init__(
            transforms=[
                UT.NormalizeFeatures(normalize_features),
                UT.SVDFeatureReduction(svd_out_dim),
                GT.GDC() if use_gdc else GT.GCNNorm(),
            ]
        )
