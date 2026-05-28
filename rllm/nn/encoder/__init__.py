from typing import TYPE_CHECKING

from rllm.utils.lazy_imports import define_lazy_imports


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

__all__, __getattr__, __dir__ = define_lazy_imports(
    __name__,
    globals(),
    _LAZY_MODULES,
)
