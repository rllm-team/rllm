from __future__ import annotations
from typing import Any, Dict, List

from rllm.types import ColType
from rllm.transforms.table_transforms import StackNumerical, TableTransform


class TabTransformerTransform(TableTransform):
    def __init__(
        self,
        out_dim: int = None,
        metadata: Dict[ColType, List[Dict[str, Any]]] = None,
    ) -> None:
        super().__init__(
            out_dim=out_dim,
            transforms=[StackNumerical(out_dim)],
        )
        self.metadata = metadata

    def reset_parameters(self) -> None:
        super().reset_parameters()
        for transform in self.transforms:
            transform.reset_parameters()
