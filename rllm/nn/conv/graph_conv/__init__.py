from typing import TYPE_CHECKING

from rllm.utils.lazy_imports import define_lazy_imports


if TYPE_CHECKING:
    from rllm.nn.conv.graph_conv.message_passing import MessagePassing
    from rllm.nn.conv.graph_conv.gat_conv import GATConv
    from rllm.nn.conv.graph_conv.gcn_conv import GCNConv
    from rllm.nn.conv.graph_conv.han_conv import HANConv
    from rllm.nn.conv.graph_conv.hgt_conv import HGTConv
    from rllm.nn.conv.graph_conv.sage_conv import SAGEConv
    from rllm.nn.conv.graph_conv.lgc_conv import LGCConv
    from rllm.nn.conv.graph_conv.transformer_conv import GTransformerConv
    from rllm.nn.conv.graph_conv.relgnn_conv import RelGNNConv
    from rllm.nn.conv.graph_conv.aggrs import (
        Aggregator,
        MeanAggregator,
        MaxAggregator,
        MinAggregator,
        SumAggregator,
        AddAggregator,
        ProdAggregator,
        GCNAggregator,
        MaxPoolAggregator,
        MeanPoolAggregator,
        LSTMAggregator,
    )

_LAZY_MODULES = {
    "rllm.nn.conv.graph_conv.message_passing": (
        "MessagePassing",
    ),
    "rllm.nn.conv.graph_conv.gat_conv": (
        "GATConv",
    ),
    "rllm.nn.conv.graph_conv.gcn_conv": (
        "GCNConv",
    ),
    "rllm.nn.conv.graph_conv.han_conv": (
        "HANConv",
    ),
    "rllm.nn.conv.graph_conv.hgt_conv": (
        "HGTConv",
    ),
    "rllm.nn.conv.graph_conv.sage_conv": (
        "SAGEConv",
    ),
    "rllm.nn.conv.graph_conv.lgc_conv": (
        "LGCConv",
    ),
    "rllm.nn.conv.graph_conv.transformer_conv": (
        "GTransformerConv",
    ),
    "rllm.nn.conv.graph_conv.relgnn_conv": (
        "RelGNNConv",
    ),
    "rllm.nn.conv.graph_conv.aggrs": (
        "Aggregator",
        "MeanAggregator",
        "MaxAggregator",
        "MinAggregator",
        "SumAggregator",
        "AddAggregator",
        "ProdAggregator",
        "GCNAggregator",
        "MaxPoolAggregator",
        "MeanPoolAggregator",
        "LSTMAggregator",
    ),
}

__all__, __getattr__, __dir__ = define_lazy_imports(
    __name__,
    globals(),
    _LAZY_MODULES,
)
