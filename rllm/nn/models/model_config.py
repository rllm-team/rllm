from rllm.nn.conv.graph_conv import GATConv
from rllm.nn.conv.graph_conv import GCNConv
from rllm.nn.conv.table_conv import ExcelFormerConv
from rllm.nn.conv.table_conv import TromptConv
from rllm.nn.conv.table_conv import TabTransformerConv
from rllm.nn.conv.table_conv import FTTransformerConv

from rllm.nn.models.tabnet import TabNet
from rllm.nn.models.rect import RECT_L

from rllm.transforms.graph_transforms import GCNTransform
from rllm.transforms.graph_transforms import RECTTransform
from rllm.transforms.table_transforms import FTTransformerTransform
from rllm.transforms.table_transforms import TabTransformerTransform
from rllm.transforms.table_transforms import TabNetTransform


MODEL_CONFIG = {
    # GNN models
    GCNConv: GCNTransform,
    GATConv: GCNTransform,
    RECT_L: RECTTransform,
    # TNN models
    TabTransformerConv: TabTransformerTransform,
    FTTransformerConv: FTTransformerTransform,
    TabNet: TabNetTransform,
    ExcelFormerConv: FTTransformerTransform,
    TromptConv: FTTransformerTransform,
}
