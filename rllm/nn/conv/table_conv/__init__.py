from typing import TYPE_CHECKING

from rllm.utils.lazy_imports import define_lazy_imports


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

__all__, __getattr__, __dir__ = define_lazy_imports(
    __name__,
    globals(),
    _LAZY_MODULES,
)
