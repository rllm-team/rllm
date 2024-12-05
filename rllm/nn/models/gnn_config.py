from typing import Any, Dict, Optional, Type

from rllm.nn.models import RECT_L

from rllm.nn.conv.graph_conv import GATConv
from rllm.nn.conv.graph_conv import GCNConv
from rllm.nn.conv.graph_conv import HGTConv

from rllm.transforms.graph_transforms import GCNTransform
from rllm.transforms.graph_transforms import RECTTransform


# Define GNN configuration dictionary
GNN_CONFIG: Dict[str, Dict[str, Any]] = {
    "GCN": {
        "conv": GCNConv,
        "transform": GCNTransform,
    },
    "GAT": {
        "conv": GATConv,
        "transform": GCNTransform,
    },
    "OGC": {
        "transform": GCNTransform,
    },
    "RECT": {
        "model": RECT_L,
        "conv": None,
        "transform": RECTTransform,
    },
    "HGT": {
        "conv": HGTConv,
        "transform": None,
    },
}


class GNNConfig:
    @classmethod
    def get_transform(cls, model_name: str) -> Optional[Type[Any]]:
        """Get the transform for a given GNN model name.

        Args:
            model_name (str): The name of the GNN model.

        Returns:
            Optional[Type[Any]]: The transform class, or None if not found.
        """
        config = GNN_CONFIG.get(model_name)
        return config["transform"] if config else None

    @classmethod
    def get_conv(cls, model_name: str) -> Optional[Type[Any]]:
        """Get the conv for a given GNN model name.

        Args:
            model_name (str): The name of the GNN model.

        Returns:
            Optional[Type[Any]]: The conv class, or None if not found.
        """
        config = GNN_CONFIG.get(model_name)
        return config["conv"] if config else None
