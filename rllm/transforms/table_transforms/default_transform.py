from __future__ import annotations
from typing import Any, Dict, List

from rllm.types import ColType
from rllm.transforms.table_transforms import TableTransform


class DefaultTransform(TableTransform):
    r"""Default Table Transform. Only fill the Nan values."""

    def __init__(
        self,
        out_dim: int = None,
        metadata: Dict[ColType, List[Dict[str, Any]]] = None,
    ) -> None:
        super().__init__(
            out_dim=out_dim,
            transforms=[],
        )
        self.metadata = metadata

    def reset_parameters(self):
        super().reset_parameters()
