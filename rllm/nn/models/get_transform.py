from typing import Any, Dict, Optional, Type

from rllm.nn.conv.graph_conv import GATConv
from rllm.nn.conv.graph_conv import GCNConv
from rllm.nn.conv.table_conv import ExcelFormerConv
from rllm.nn.conv.table_conv import TromptConv
from rllm.nn.conv.table_conv import TabTransformerConv
from rllm.nn.conv.table_conv import FTTransformerConv

from rllm.nn.models import RECT_L
from rllm.nn.models import TabNet

from rllm.transforms.graph_transforms import GCNTransform
from rllm.transforms.graph_transforms import RECTTransform
from rllm.transforms.table_transforms import FTTransformerTransform
from rllm.transforms.table_transforms import TabTransformerTransform
from rllm.transforms.table_transforms import TabNetTransform


# Define GNN configuration dictionary
GNN_CONV_TO_TRANSFORM: Dict[Type[Any], Type[Any]] = {
    GCNConv: GCNTransform,
    GATConv: GCNTransform,
    RECT_L: RECTTransform,
}

# Define TNN configuration dictionary
TNN_CONV_TO_TRANSFORM: Dict[Type[Any], Type[Any]] = {
    TabTransformerConv: TabTransformerTransform,
    FTTransformerConv: FTTransformerTransform,
    ExcelFormerConv: FTTransformerTransform,
    TromptConv: FTTransformerTransform,
    TabNet: TabNetTransform,
}


def get_transform(conv: Type[Any]) -> Optional[Type[Any]]:
    """Get the default transform for a given conv class.

    Args:
        conv (Type[Any]): The conv class.

    Returns:
        Optional[Type[Any]]: The default transform class, or None if not found.
    """
    if conv in GNN_CONV_TO_TRANSFORM:
        return GNN_CONV_TO_TRANSFORM[conv]
    elif conv in TNN_CONV_TO_TRANSFORM:
        return TNN_CONV_TO_TRANSFORM[conv]
    else:
        return None
