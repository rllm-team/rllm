from typing import TYPE_CHECKING

from rllm.utils.lazy_imports import define_lazy_imports


if TYPE_CHECKING:
    from rllm.transforms.table_transforms.col_transform import ColTransform
    from rllm.transforms.table_transforms.col_normalize import ColNormalize
    from rllm.transforms.table_transforms.one_hot_transform import OneHotTransform
    from rllm.transforms.table_transforms.stack_numerical import StackNumerical
    from rllm.transforms.table_transforms.table_transform import TableTransform
    from rllm.transforms.table_transforms.tab_transformer_transform import TabTransformerTransform
    from rllm.transforms.table_transforms.default_table_transform import DefaultTableTransform

_LAZY_MODULES = {
    "rllm.transforms.table_transforms.col_transform": (
        "ColTransform",
    ),
    "rllm.transforms.table_transforms.col_normalize": (
        "ColNormalize",
    ),
    "rllm.transforms.table_transforms.one_hot_transform": (
        "OneHotTransform",
    ),
    "rllm.transforms.table_transforms.stack_numerical": (
        "StackNumerical",
    ),
    "rllm.transforms.table_transforms.table_transform": (
        "TableTransform",
    ),
    "rllm.transforms.table_transforms.tab_transformer_transform": (
        "TabTransformerTransform",
    ),
    "rllm.transforms.table_transforms.default_table_transform": (
        "DefaultTableTransform",
    ),
}

__all__, __getattr__, __dir__ = define_lazy_imports(
    __name__,
    globals(),
    _LAZY_MODULES,
)
