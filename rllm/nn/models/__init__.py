from importlib import import_module
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from rllm.nn.models.rect import RECT_L
    from rllm.nn.models.bridge import (
        BRIDGE,
        TableEncoder,
        GraphEncoder,
    )
    from rllm.nn.models.transtab import (
        TransTab,
        TransTabClassifier,
        TransTabForCL,
    )
    from rllm.nn.models.resnet import TableResNet
    from rllm.nn.models.heterosage import HeteroSAGE
    from rllm.nn.models.rdl import RDL
    from rllm.nn.models.relgnn import (
        RelGNN,
        RelGNNModel,
    )

_LAZY_MODULES = {
    "rllm.nn.models.rect": (
        "RECT_L",
    ),
    "rllm.nn.models.bridge": (
        "BRIDGE",
        "TableEncoder",
        "GraphEncoder",
    ),
    "rllm.nn.models.transtab": (
        "TransTab",
        "TransTabClassifier",
        "TransTabForCL",
    ),
    "rllm.nn.models.resnet": (
        "TableResNet",
    ),
    "rllm.nn.models.heterosage": (
        "HeteroSAGE",
    ),
    "rllm.nn.models.rdl": (
        "RDL",
    ),
    "rllm.nn.models.relgnn": (
        "RelGNN",
        "RelGNNModel",
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
