from importlib import import_module
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from rllm.utils.download import (
        download_url,
        download_google_url,
    )
    from rllm.utils.extract import extract_zip
    from rllm.utils.sparse import (
        sparse_mx_to_torch_sparse_tensor,
        is_torch_sparse_tensor,
        get_indices,
        set_values,
    )
    from rllm.utils.undirected import (
        is_undirected,
        to_undirected,
    )
    from rllm.utils.seg_reduce import (
        seg_sum,
        seg_softmax,
        seg_softmax_,
    )
    from rllm.utils.graph_utils import (
        adj_to_edge_index,
        sort_edge_index,
        index_to_ptr,
        _to_csc,
    )
    from rllm.utils._sort import lexsort
    from rllm.utils._remap import remap_keys
    from rllm.utils._mixin import CastMixin
    from rllm.utils.atomic_routes import get_atomic_routes

_LAZY_MODULES = {
    "rllm.utils.download": (
        "download_url",
        "download_google_url",
    ),
    "rllm.utils.extract": (
        "extract_zip",
    ),
    "rllm.utils.sparse": (
        "sparse_mx_to_torch_sparse_tensor",
        "is_torch_sparse_tensor",
        "get_indices",
        "set_values",
    ),
    "rllm.utils.undirected": (
        "is_undirected",
        "to_undirected",
    ),
    "rllm.utils.seg_reduce": (
        "seg_sum",
        "seg_softmax",
        "seg_softmax_",
    ),
    "rllm.utils.graph_utils": (
        "adj_to_edge_index",
        "sort_edge_index",
        "index_to_ptr",
        "_to_csc",
    ),
    "rllm.utils._sort": (
        "lexsort",
    ),
    "rllm.utils._remap": (
        "remap_keys",
    ),
    "rllm.utils._mixin": (
        "CastMixin",
    ),
    "rllm.utils.atomic_routes": (
        "get_atomic_routes",
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
