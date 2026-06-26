from typing import TYPE_CHECKING

from rllm.utils.lazy_imports import define_lazy_imports


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
    from rllm.nn.models.metartl import (
        MetaPathFusion,
        MetaPathProp,
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
    "rllm.nn.models.metartl": (
        "MetaPathFusion",
        "MetaPathProp",
    ),
}

__all__, __getattr__, __dir__ = define_lazy_imports(
    __name__,
    globals(),
    _LAZY_MODULES,
)
