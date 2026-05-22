from importlib import import_module
from typing import TYPE_CHECKING


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
