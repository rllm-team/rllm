from __future__ import annotations
from collections import Counter

import torch
from enum import Enum
from torch import Tensor


class TableType(Enum):
    r"""The semantic type of a table.

    .. code-block:: python

        from rllm.types import TableType

        table_type = TableType.DATATABLE  # data table
        table_type = TableType.RELATIONSHIPTABLE  # relationship table
        ...

    Attributes:
        DATATABLE: Data table.
        RELATIONSHIPTABLE: Relationship table.
    """

    DATATABLE = "dataTable"
    RELATIONSHIPTABLE = "relationshipTable"


class ColType(Enum):
    r"""The semantic type of a column.

    A semantic type denotes the semantic meaning of a column, and denotes how
    columns are encoded into an embedding space within tabular deep learning
    models:

    .. code-block:: python

        from rllm.types import ColType

        col_type = ColType.NUMERICAL  # Numerical columns
        col_type = ColType.CATEGORICAL  # Categorical columns
        ...

    Attributes:
        NUMERICAL: Numerical columns.
        CATEGORICAL: Categorical columns.
    """

    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"

    def __lt__(self, other):
        return self.value < other.value


class TaskType(Enum):
    r"""The semantic type of a task.

    Attributes:
        REGRESSION: Regression task.
        MULTI_CLASSIFICATION: Multi-class classification task.
        BINARY_CLASSIFICATION: Binary classification task.
        MULTILABEL_CLASSIFICATION: Multi-label classification task.
    """

    REGRESSION = "regression"
    MULTI_CLASSIFICATION = "multiclass_classification"
    BINARY_CLASSIFICATION = "binary_classification"
    # TODO: support multi-label
    MULTILABEL_CLASSIFICATION = "multilabel_classification"


class NAMode(Enum):
    r"""The semantic type of how to process na value.

    Attributes:
        MOST_FREQUENT: Use most frequent number in column to replace nan.
        MAX: Use max number in column to replace nan.
        MIN: Use min number in column to replace nan.
        MEAN: Use mean of column to replace nan.
    """

    MOST_FREQUENT = "most_frequent"
    MAX = "max"
    MIN = "min"
    MEAN = "mean"
    ZERO = "zero"

    @staticmethod
    def namode_for_col_type(col_type: ColType) -> list[ColType]:
        namode_type = {
            ColType.NUMERICAL: [NAMode.MAX, NAMode.MIN, NAMode.MEAN, NAMode.ZERO],
            ColType.CATEGORICAL: [NAMode.MOST_FREQUENT, NAMode.ZERO],
        }
        return namode_type.get(col_type, [])


class StatType(Enum):
    r"""The different types for column statistics.

    Attributes:
        MEAN: The average value of a numerical column.
        MAX: The max value of a numerical column.
        MIN: The min value of a numerical column.
        STD: The standard deviation of a numerical column.
        QUANTILES: The minimum, first quartile, median, third quartile,
            and the maximum of a numerical column.
        COUNT: The unique category count of each category in a
            categorical column.
        MOST_FREQUENT: The most frequent catrgory in a categorical column.
    """

    # Column name
    COLNAME = "COLNAME"

    # Numerical:
    MEAN = "MEAN"
    MAX = "MAX"
    MIN = "MIN"
    STD = "STD"
    QUANTILES = "QUANTILES"

    # categorical:
    COUNT = "COUNT"
    MOST_FREQUENT = "MOST_FREQUENT"

    @staticmethod
    def stats_for_col_type(col_type: ColType) -> list[ColType]:
        stats_type = {
            ColType.NUMERICAL: [
                StatType.MEAN,
                StatType.MAX,
                StatType.MIN,
                StatType.STD,
                StatType.QUANTILES,
            ],
            ColType.CATEGORICAL: [
                StatType.COUNT,
                StatType.MOST_FREQUENT,
            ],
        }
        return stats_type.get(col_type, [])

    @staticmethod
    def compute(tensor: Tensor, stat_type: StatType):
        # stat_type for numerical
        if stat_type == StatType.MEAN:
            return torch.mean(tensor[~torch.isnan(tensor)]).item()
        if stat_type == StatType.MAX:
            return torch.max(tensor[~torch.isnan(tensor)]).item()
        if stat_type == StatType.MIN:
            return torch.min(tensor[~torch.isnan(tensor)]).item()
        if stat_type == StatType.STD:
            return torch.std(tensor[~torch.isnan(tensor)]).item()
        if stat_type == StatType.QUANTILES:
            return [
                torch.quantile(tensor[~torch.isnan(tensor)], 0).item(),
                torch.quantile(tensor[~torch.isnan(tensor)], 0.25).item(),
                torch.quantile(tensor[~torch.isnan(tensor)], 0.5).item(),
                torch.quantile(tensor[~torch.isnan(tensor)], 0.75).item(),
                torch.quantile(tensor[~torch.isnan(tensor)], 1).item(),
            ]

        # stat_type for categorical
        if stat_type == StatType.COUNT:
            return int(torch.max(tensor[tensor != -1]).item() + 1)
        if stat_type == StatType.MOST_FREQUENT:
            counter = Counter(tensor[tensor != -1].tolist())
            return int(max(counter, key=counter.get))
