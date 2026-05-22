from typing import TYPE_CHECKING

from rllm.utils.lazy_imports import define_lazy_imports


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

__all__, __getattr__, __dir__ = define_lazy_imports(
    __name__,
    globals(),
    _LAZY_MODULES,
)
