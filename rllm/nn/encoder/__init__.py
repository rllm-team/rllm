from importlib import import_module
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from rllm.nn.encoder.table_pre_encoder import TablePreEncoder
    from rllm.nn.encoder.ft_transformer_pre_encoder import FTTransformerPreEncoder
    from rllm.nn.encoder.tab_transformer_pre_encoder import TabTransformerPreEncoder
    from rllm.nn.encoder.transtab_pre_encoder import TransTabPreEncoder
    from rllm.nn.encoder.resnet_pre_encoder import ResNetPreEncoder
    from rllm.nn.encoder.trompt_pre_encoder import TromptPreEncoder
    from rllm.nn.encoder.heterotemporal_encoder import HeteroTemporalEncoder

_LAZY_MODULES = {
    "rllm.nn.encoder.table_pre_encoder": (
        "TablePreEncoder",
    ),
    "rllm.nn.encoder.ft_transformer_pre_encoder": (
        "FTTransformerPreEncoder",
    ),
    "rllm.nn.encoder.tab_transformer_pre_encoder": (
        "TabTransformerPreEncoder",
    ),
    "rllm.nn.encoder.transtab_pre_encoder": (
        "TransTabPreEncoder",
    ),
    "rllm.nn.encoder.resnet_pre_encoder": (
        "ResNetPreEncoder",
    ),
    "rllm.nn.encoder.trompt_pre_encoder": (
        "TromptPreEncoder",
    ),
    "rllm.nn.encoder.heterotemporal_encoder": (
        "HeteroTemporalEncoder",
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
