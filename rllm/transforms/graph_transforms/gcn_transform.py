import rllm.transforms.graph_transforms as GT
import rllm.transforms.utils as UT


class GCNTransform(GT.GraphTransform):
    def __init__(self, normalize_features: str = "l1"):
        super().__init__(
            transforms=[
                UT.NormalizeFeatures(normalize_features),
                GT.GCNNorm(),
            ]
        )
