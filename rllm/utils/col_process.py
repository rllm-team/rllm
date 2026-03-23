import numpy as np
import pandas as pd


def timecol_to_unix_time(ser: pd.Series) -> np.ndarray:
    r"""Convert a :class:`pandas.Series` of datetime values to UNIX
    timestamps in seconds.

    Args:
        ser (pandas.Series): A datetime series with dtype
            :obj:`datetime64[s]` or :obj:`datetime64[ns]`.

    Returns:
        numpy.ndarray: An array of integer UNIX timestamps (seconds).
    """
    assert ser.dtype in [np.dtype("datetime64[s]"), np.dtype("datetime64[ns]")]
    unix_time = ser.astype("int64").values
    if ser.dtype == np.dtype("datetime64[ns]"):
        unix_time //= 10**9
    return unix_time
