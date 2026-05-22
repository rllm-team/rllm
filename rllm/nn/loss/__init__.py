from typing import TYPE_CHECKING

from rllm.utils.lazy_imports import define_lazy_imports


if TYPE_CHECKING:
    from rllm.nn.loss.base_loss import BaseLoss
    from rllm.nn.loss.contrastive_loss import ContrastiveLoss
    from rllm.nn.loss.vpcl_loss import (
        SelfSupervisedVPCL,
        SupervisedVPCL,
    )

_LAZY_MODULES = {
    "rllm.nn.loss.base_loss": (
        "BaseLoss",
    ),
    "rllm.nn.loss.contrastive_loss": (
        "ContrastiveLoss",
    ),
    "rllm.nn.loss.vpcl_loss": (
        "SelfSupervisedVPCL",
        "SupervisedVPCL",
    ),
}

__all__, __getattr__, __dir__ = define_lazy_imports(
    __name__,
    globals(),
    _LAZY_MODULES,
)
