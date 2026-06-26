from typing import TYPE_CHECKING

from rllm.utils.lazy_imports import define_lazy_imports


if TYPE_CHECKING:
    from rllm.nn.models.metartl.linear_per_metapath import LinearPerMetapath
    from rllm.nn.models.metartl.semantic_transformer import SemanticTransformer
    from rllm.nn.models.metartl.metapath_fusion import MetaPathFusion
    from rllm.nn.models.metartl.metapath_prop_transform import MetaPathProp

_LAZY_MODULES = {
    "rllm.nn.models.metartl.linear_per_metapath": (
        "LinearPerMetapath",
    ),
    "rllm.nn.models.metartl.semantic_transformer": (
        "SemanticTransformer",
    ),
    "rllm.nn.models.metartl.metapath_fusion": (
        "MetaPathFusion",
    ),
    "rllm.nn.models.metartl.metapath_prop_transform": (
        "MetaPathProp",
    ),
}

__all__, __getattr__, __dir__ = define_lazy_imports(
    __name__,
    globals(),
    _LAZY_MODULES,
)
