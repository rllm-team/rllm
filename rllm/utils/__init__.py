from typing import TYPE_CHECKING

from rllm.utils.lazy_imports import define_lazy_imports


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
    from rllm.utils.xavier_init import _xavier_uniform_

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
    "rllm.utils.xavier_init": (
        "_xavier_uniform_",
    ),
}

__all__, __getattr__, __dir__ = define_lazy_imports(
    __name__,
    globals(),
    _LAZY_MODULES,
)
