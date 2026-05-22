from importlib import import_module
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from rllm.data.graph_data import (
        BaseGraph,
        GraphData,
        HeteroGraphData,
    )
    from rllm.data.table_data import (
        BaseTable,
        TableData,
        TableDataset,
        TextEmbedderConfig,
    )
    from rllm.data.storage import (
        BaseStorage,
        NodeStorage,
        EdgeStorage,
        recursive_apply,
    )
    from rllm.data.view import (
        MappingView,
        KeysView,
        ValuesView,
        ItemsView,
    )

_LAZY_MODULES = {
    "rllm.data.graph_data": (
        "BaseGraph",
        "GraphData",
        "HeteroGraphData",
    ),
    "rllm.data.table_data": (
        "BaseTable",
        "TableData",
        "TableDataset",
        "TextEmbedderConfig",
    ),
    "rllm.data.storage": (
        "BaseStorage",
        "NodeStorage",
        "EdgeStorage",
        "recursive_apply",
    ),
    "rllm.data.view": (
        "MappingView",
        "KeysView",
        "ValuesView",
        "ItemsView",
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
