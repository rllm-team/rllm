from importlib import import_module
from typing import TYPE_CHECKING


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

__all__ = [
    name
    for names in _LAZY_MODULES.values()
    for name in names
]

_LAZY_ATTRS = {
    name: module_name
    for module_name, names in _LAZY_MODULES.items()
    for name in names
}


def __getattr__(name):
    if name in _LAZY_ATTRS:
        module = import_module(_LAZY_ATTRS[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
