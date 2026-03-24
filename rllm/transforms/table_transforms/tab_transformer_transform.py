from __future__ import annotations
from typing import Any, Dict, List

from rllm.types import ColType
from rllm.transforms.table_transforms import TableTransform,ColNormalize


class TabTransformerTransform(TableTransform):
    r"""TabTransformerTransform applies ColNormalize and StackNumerical
    transform to tabular data specifically for the TabTransformer model.

    Args:
        out_dim (int): The output dimensionality.
        metadata (Dict[ColType, List[Dict[str, Any]]], optional): Metadata
            containing information about the columns, such as statistics.
            (default: :obj:`None`)

    Examples:
        >>> transform = TabTransformerTransform(out_dim=32)
        >>> data = transform(data)
    """

    def __init__(
        self,
        out_dim: int,
        metadata: Dict[ColType, List[Dict[str, Any]]] = None,
    ) -> None:
        super().__init__(
            out_dim=out_dim, transforms=[ColNormalize()]
        )
        self.metadata = metadata

    def reset_parameters(self) -> None:
        super().reset_parameters()
        for transform in self.transforms:
            if hasattr(transform, "reset_parameters"):
                transform.reset_parameters()
