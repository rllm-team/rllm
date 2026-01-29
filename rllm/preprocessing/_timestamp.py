from typing import Optional

from pandas import Series
import pandas as pd
import torch
from torch import Tensor


class TimestampPreprocessor:
    r"""Maps a timestamp df column to tensor."""

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

    def __init__(self, format: str):
        super().__init__()
        self.format = format

    @staticmethod
    def to_tensor(ser: Series) -> Tensor:
        # subtracting 1 so that the smallest months and days can
        # start from 0.
        tensors = [
            torch.from_numpy(ser.dt.year.values).unsqueeze(1),
            torch.from_numpy(ser.dt.month.values - 1).unsqueeze(1),
            torch.from_numpy(ser.dt.day.values - 1).unsqueeze(1),
            torch.from_numpy(ser.dt.dayofweek.values).unsqueeze(1),
            torch.from_numpy(ser.dt.hour.values).unsqueeze(1),
            torch.from_numpy(ser.dt.minute.values).unsqueeze(1),
            torch.from_numpy(ser.dt.second.values).unsqueeze(1),
        ]
        stacked = torch.cat(tensors, dim=1)
        return torch.nan_to_num(stacked, nan=-1).to(torch.long)

    def __call__(
        self,
        ser: Series,
        *,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        ser = pd.to_datetime(ser, format=self.format, errors="coerce")
        tensor = TimestampPreprocessor.to_tensor(ser)
        return tensor.to(device)
