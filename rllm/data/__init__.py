from typing import TYPE_CHECKING

from rllm.utils.lazy_imports import define_lazy_imports


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

__all__, __getattr__, __dir__ = define_lazy_imports(
    __name__,
    globals(),
    _LAZY_MODULES,
)
