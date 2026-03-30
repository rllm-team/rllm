from __future__ import annotations
from typing import Any, Dict, List, Optional

from rllm.types import ColType
from rllm.transforms.table_transforms import TableTransform


class DefaultTableTransform(TableTransform):
    r"""Default table transform that only performs missing-value handling.

    Args:
        out_dim (int): The output dimensionality.
        metadata (Dict[ColType, List[Dict[str, Any]]], optional): Metadata
            containing information about the columns, such as statistics.
            (default: :obj:`None`)
    """

    def __init__(
        self,
        out_dim: Optional[int] = None,
        metadata: Optional[Dict[ColType, List[Dict[str, Any]]]] = None,
    ) -> None:
        super().__init__(
            out_dim=out_dim,
            transforms=[],
        )
        self.metadata = metadata

    def reset_parameters(self):
        super().reset_parameters()
