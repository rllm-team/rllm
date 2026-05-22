from importlib import import_module
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from rllm.nn.conv.table_conv.ft_transformer_conv import FTTransformerConv
    from rllm.nn.conv.table_conv.tab_transformer_conv import TabTransformerConv
    from rllm.nn.conv.table_conv.excelformer_conv import ExcelFormerConv
    from rllm.nn.conv.table_conv.trompt_conv import TromptConv
    from rllm.nn.conv.table_conv.saint_conv import SAINTConv
    from rllm.nn.conv.table_conv.transtab_conv import TransTabConv
    from rllm.nn.conv.table_conv.resnet_conv import ResNetConv

_LAZY_MODULES = {
    "rllm.nn.conv.table_conv.ft_transformer_conv": (
        "FTTransformerConv",
    ),
    "rllm.nn.conv.table_conv.tab_transformer_conv": (
        "TabTransformerConv",
    ),
    "rllm.nn.conv.table_conv.excelformer_conv": (
        "ExcelFormerConv",
    ),
    "rllm.nn.conv.table_conv.trompt_conv": (
        "TromptConv",
    ),
    "rllm.nn.conv.table_conv.saint_conv": (
        "SAINTConv",
    ),
    "rllm.nn.conv.table_conv.transtab_conv": (
        "TransTabConv",
    ),
    "rllm.nn.conv.table_conv.resnet_conv": (
        "ResNetConv",
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
