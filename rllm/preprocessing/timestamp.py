from typing import List, Optional, Sequence

from pandas import Series
import pandas as pd
import torch
from torch import Tensor

class TimestampPreprocessor:
    r"""Convert a timestamp column to a tensor.

    Datetime values are split into integer time components (year, month,
    day, etc.). By default all seven components are used and the output
    has shape ``[N, 7]``. Missing / unparseable values become ``-1``.
    """

    # Supported time field names, in default extraction order.
    ALL_FIELDS: List[str] = ["YEAR", "MONTH", "DAY", "DAYOFWEEK", "HOUR", "MINUTE", "SECOND"]

    NUM_MONTHS_PER_YEAR = 12
    NUM_DAYS_PER_MONTH = 31  # approximate
    NUM_DAYS_PER_WEEK = 7
    NUM_HOURS_PER_DAY = 24
    NUM_MINUTES_PER_HOUR = 60
    NUM_SECONDS_PER_MINUTE = 60

    TIME_TO_INDEX = {
        "YEAR": 0,
        "MONTH": 1,
        "DAY": 2,
        "DAYOFWEEK": 3,
        "HOUR": 4,
        "MINUTE": 5,
        "SECOND": 6,
    }

    # Cyclic normalisation constants for MONTH ~ SECOND (excludes YEAR).
    CYCLIC_VALUES_NORMALIZATION_CONSTANT = torch.tensor(
        [
            NUM_MONTHS_PER_YEAR,
            NUM_DAYS_PER_MONTH,
            NUM_DAYS_PER_WEEK,
            NUM_HOURS_PER_DAY,
            NUM_MINUTES_PER_HOUR,
            NUM_SECONDS_PER_MINUTE,
        ]
    )

    # Map each field name to a callable (Series → numpy array).
    _FIELD_EXTRACTORS = {
        "YEAR":      lambda s: s.dt.year.values,
        "MONTH":     lambda s: s.dt.month.values - 1,
        "DAY":       lambda s: s.dt.day.values - 1,
        "DAYOFWEEK": lambda s: s.dt.dayofweek.values,
        "HOUR":      lambda s: s.dt.hour.values,
        "MINUTE":    lambda s: s.dt.minute.values,
        "SECOND":    lambda s: s.dt.second.values,
    }

    def __init__(
        self,
        format: Optional[str] = None,
        fields: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.format = format

        if fields is None:
            self.fields: List[str] = list(TimestampPreprocessor.ALL_FIELDS)
        else:
            invalid = [f for f in fields if f not in self.TIME_TO_INDEX]
            if invalid:
                raise ValueError(
                    f"Unknown field(s) {invalid}. "
                    f"Valid fields are: {list(self.TIME_TO_INDEX.keys())}."
                )
            self.fields = list(fields)

    @staticmethod
    def to_tensor(
        ser: Series,
        fields: Optional[Sequence[str]] = None,
    ) -> Tensor:
        """Convert a ``datetime64`` Series to a long tensor.

        Args:
            ser: Series with ``dtype=datetime64[ns]``.
            fields: Ordered list of field names to extract.  Defaults to
                ``None`` which extracts all seven fields.

        Returns:
            Tensor of shape ``[N, len(fields)]`` with ``dtype=torch.long``.
            Missing values are encoded as ``-1``.
        """
        if fields is None:
            fields = TimestampPreprocessor.ALL_FIELDS

        tensors = [
            torch.from_numpy(
                TimestampPreprocessor._FIELD_EXTRACTORS[f](ser)
            ).unsqueeze(1)
            for f in fields
        ]
        stacked = torch.cat(tensors, dim=1)
        return torch.nan_to_num(stacked, nan=-1).to(torch.long)

    def __call__(
        self,
        ser: Series,
        *,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """Parse ``ser`` and return the time-component tensor.

        Args:
            ser: Raw timestamp column (string or already datetime).
            device: Target PyTorch device.  Defaults to ``None`` (CPU).

        Returns:
            Tensor of shape ``[N, len(self.fields)]``.
        """
        ser = pd.to_datetime(ser, format=self.format, errors="coerce")
        tensor = TimestampPreprocessor.to_tensor(ser, fields=self.fields)
        return tensor.to(device)
