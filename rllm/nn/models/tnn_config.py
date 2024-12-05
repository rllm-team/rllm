from typing import Dict, Any, Type, Optional

from rllm.nn.conv.table_conv import ExcelFormerConv
from rllm.nn.conv.table_conv import TromptConv
from rllm.nn.conv.table_conv import TabTransformerConv
from rllm.nn.conv.table_conv import FTTransformerConv

from rllm.nn.models import TabNet

from rllm.transforms.table_transforms import TabTransformerTransform
from rllm.transforms.table_transforms import DefaultTransform

from rllm.nn.pre_encoder import TabTransformerEncoder
from rllm.nn.pre_encoder import FTTransformerEncoder

TNN_CONFIG: Dict[str, Dict[str, Any]] = {
    "TabTransformer": {
        "conv": TabTransformerConv,
        "transform": TabTransformerTransform,
        "pre_encoder": TabTransformerEncoder,
    },
    "FTTransformer": {
        "conv": FTTransformerConv,
        "transform": DefaultTransform,
        "pre_encoder": FTTransformerEncoder,
    },
    "ExcelFormer": {
        "conv": ExcelFormerConv,
        "transform": DefaultTransform,
        "pre_encoder": FTTransformerEncoder,
    },
    "Trompt": {
        "conv": TromptConv,
        "transform": DefaultTransform,
        "pre_encoder": FTTransformerEncoder,
    },
    "TabNet": {
        "model": TabNet,
        "conv": None,
        "transform": DefaultTransform,
        "pre_encoder": FTTransformerEncoder,
    },
}


class TNNConfig:
    @classmethod
    def get_transform(cls, model_name: str) -> Optional[Type[Any]]:
        """Get the transform for a given TNN model name.

        Args:
            model_name (str): The name of the TNN model.

        Returns:
            Optional[Type[Any]]: The transform class, or None if not found.
        """
        config = TNN_CONFIG.get(model_name)
        return config["transform"] if config else None

    @classmethod
    def get_conv(cls, model_name: str) -> Optional[Type[Any]]:
        """Get the conv for a given TNN model name.

        Args:
            model_name (str): The name of the TNN model.

        Returns:
            Optional[Type[Any]]: The conv class, or None if not found.
        """
        config = TNN_CONFIG.get(model_name)
        return config["conv"] if config else None

    @classmethod
    def get_pre_encoder(cls, model_name: str) -> Optional[Type[Any]]:
        """Get the pre_encoder for a given TNN model name.

        Args:
            model_name (str): The name of the TNN model.

        Returns:
            Optional[Type[Any]]: The pre_encoder class, or None if not found.
        """
        config = TNN_CONFIG.get(model_name)
        return config["pre_encoder"] if config else None
