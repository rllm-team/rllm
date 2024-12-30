from __future__ import annotations
from typing import Any, Dict, List

from rllm.transforms.table_transforms.column_normalize import ColumnNormalize
from rllm.types import ColType
from rllm.transforms.table_transforms import StackNumerical, TableTransform


class TabTransformerTransform(TableTransform):
    r"""TabTransformerTransform applies ColumnNormalize and StackNumerical
    transform to tabular data specifically for the TabTransformer model.

    Args:
        out_dim (int): The output dimensionality.
        metadata (Dict[ColType, List[Dict[str, Any]]], optional): Metadata
            containing information about the columns, such as statistics.
            (default: :obj:`None`)
    """

    def __init__(
        self,
        out_dim: int,
        metadata: Dict[ColType, List[Dict[str, Any]]] = None,
    ) -> None:
        super().__init__(
            out_dim=out_dim,
            transforms=[ColumnNormalize(), StackNumerical(out_dim)],
        )
        self.metadata = metadata

    def reset_parameters(self) -> None:
        super().reset_parameters()
        for transform in self.transforms:
            transform.reset_parameters()
