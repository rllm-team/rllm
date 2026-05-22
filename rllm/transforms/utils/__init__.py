from importlib import import_module
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from rllm.transforms.utils.base_transform import BaseTransform
    from rllm.transforms.utils.remove_training_classes import RemoveTrainingClasses

_LAZY_MODULES = {
    "rllm.transforms.utils.base_transform": (
        "BaseTransform",
    ),
    "rllm.transforms.utils.remove_training_classes": (
        "RemoveTrainingClasses",
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
