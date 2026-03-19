from typing import Dict, Any, Optional, Union
from enum import Enum
import warnings
import math

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import pandas.api.types as ptypes
from dateutil.parser import ParserError

from rllm.types import ColType


# These are additional column types beyond ColType
# Not supported for now.
class ColTypePlaceholder(Enum):
    SEQUENCE_NUMERICAL = "sequence_numerical"
    MULTICATEGORICAL = "multicategorical"
    EMBEDDING = "embedding"


class TypeInferencer:
    """A utility class for inferring column types in tabular data.

    This class provides methods to infer the semantic type of columns
    in a DataFrame based on their content.
    Inferred types might be incorrect, so please use with caution.
    """

    # Categorical minimum counting threshold. If the count of the most minor
    # categories is larger than this value, we treat the column as categorical.
    # This is the original setting in TorchFrame, I keep it for consistency
    # with RelBench.
    cat_min_count_thresh = 4

    POSSIBLE_SEPS = ["|", ","]
    POSSIBLE_TIME_FORMATS = [None, "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d"]

    @classmethod
    def _is_timestamp(cls, ser: Series) -> bool:
        is_timestamp = False
        for time_format in cls.POSSIBLE_TIME_FORMATS:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pd.to_datetime(ser, format=time_format)
                is_timestamp = True
            except (ValueError, ParserError, TypeError):
                pass
        return is_timestamp

    @classmethod
    def _lst_is_all_type(
        cls,
        lst: list[Any],
        types: Union[tuple[type, ...], type],
    ) -> bool:
        assert isinstance(lst, list)
        return all(isinstance(x, types) for x in lst)

    @classmethod
    def _lst_is_free_of_nan_and_inf(cls, lst: list[Any]):
        assert isinstance(lst, list)
        return all(not math.isnan(x) and not math.isinf(x) for x in lst)

    @classmethod
    def _min_count(cls, ser: Series) -> int:
        return ser.value_counts().min()

    @classmethod
    def infer_series_coltype(cls, ser: Series) -> Optional[ColType]:
        """Infer :obj:`ColType` given :class:`Series` object.

        Args:
            ser (Series): Input series.

        Returns:
            Optional[ColType]: Inferred :obj:`ColType`. Returns :obj:`None` if
                inference failed.
        """
        has_nan = ser.isna().any()
        if has_nan:
            ser = ser.dropna()

        if len(ser) == 0:
            return None

        if isinstance(ser.iloc[0], list):
            # Candidates: embedding, sequence_numerical, multicategorical

            # True if all elements in all lists are numerical
            is_all_numerical = True
            # True if all elements in all lists are string
            is_all_string = True
            # True if all lists are of the same length and all elements are float
            # and free of nans.
            is_embedding = True

            length = len(ser.iloc[0])
            for lst in ser:
                if not isinstance(lst, list):
                    return None
                if cls._lst_is_all_type(lst, (int, float)):
                    if not (
                        length == len(lst)
                        and cls._lst_is_all_type(lst, float)
                        and cls._lst_is_free_of_nan_and_inf(lst)
                    ):
                        is_embedding = False
                else:
                    is_all_numerical = False
                if not cls._lst_is_all_type(lst, str):
                    is_all_string = False

            if is_all_numerical:
                if is_embedding:
                    return ColTypePlaceholder.EMBEDDING
                else:
                    return ColTypePlaceholder.SEQUENCE_NUMERICAL
            elif is_all_string:
                return ColTypePlaceholder.MULTICATEGORICAL
            else:
                return None
        else:
            # Candidates: numerical, categorical, multicategorical, and
            # text_(embedded/tokenized)

            if ptypes.is_numeric_dtype(ser):

                if ptypes.is_bool_dtype(ser):
                    return ColType.BINARY
                # Candidates: numerical, categorical
                if ptypes.is_float_dtype(ser) and not (
                    has_nan and (ser % 1 == 0).all()
                ):
                    return ColType.NUMERICAL
                else:
                    if cls._min_count(ser) > cls.cat_min_count_thresh:
                        return ColType.CATEGORICAL
                    else:
                        return ColType.NUMERICAL
            else:
                # Candidates: timestamp, categorical, multicategorical,
                # text_(embedded/tokenized), embedding
                if cls._is_timestamp(ser):
                    return ColType.TIMESTAMP

                # Candates: categorical, multicategorical,
                # text_(embedded/tokenized), embedding
                if cls._min_count(ser) > cls.cat_min_count_thresh:
                    if ptypes.is_bool_dtype(ser):
                        return ColType.BINARY
                    else:
                        return ColType.CATEGORICAL

                # Candates: multicategorical, text_(embedded/tokenized), embedding
                if not ptypes.is_string_dtype(ser):
                    if cls._min_count(ser) > cls.cat_min_count_thresh:
                        return ColTypePlaceholder.MULTICATEGORICAL
                    else:
                        return ColTypePlaceholder.EMBEDDING

                # Try different possible seps and mick the largest min_count.
                if isinstance(ser.iloc[0], list) or isinstance(ser.iloc[0], np.ndarray):
                    max_min_count = cls._min_count(ser.explode())
                else:
                    min_count_list = []
                    for sep in cls.POSSIBLE_SEPS:
                        try:
                            # TODO: For now, not used
                            pass
                        except Exception as e:
                            warnings.warn(
                                "Mapping series into multicategorical stype "
                                f"with separator {sep} raised an exception {e}"
                            )
                            continue
                    max_min_count = max(min_count_list or [0])

                if max_min_count > cls.cat_min_count_thresh:
                    return ColTypePlaceholder.MULTICATEGORICAL
                else:
                    return ColType.TEXT

    @classmethod
    def infer_df_coltype(cls, df: DataFrame) -> dict[str, ColType]:
        """Infer :obj:`col_to_type` given :class:`DataFrame` object.

        Args:
            df (DataFrame): Input data frame.

        Returns:
            col_to_type: Inferred :obj:`col_to_type`, mapping a column name to
                its inferred :obj:`ColType`.
        """
        col_to_type = {}
        for col in df.columns:
            coltype = cls.infer_series_coltype(df[col])
            if coltype is not None:
                col_to_type[col] = coltype
            else:
                warnings.warn(f"Failed to infer column type for column {col}.")
        return col_to_type

    @classmethod
    def infer_table_df_dict_coltype(
        cls,
        df_dict: Dict[str, DataFrame],
    ):
        """Infer :obj:`col_to_type` for each table in a dictionary of DataFrames.

        Args:
            df_dict (Dict[str, DataFrame]): A dictionary mapping table names to
                DataFrame objects.

        Returns:
            Dict[str, dict]: A dictionary mapping table names to their inferred
                :obj:`col_to_type` dictionaries.
        """
        df_coltype_dict = {}
        for table_name, df in df_dict.items():
            df = df.sample(min(1_000, len(df)))
            inferred_col_to_coltype = cls.infer_df_coltype(df)
            df_coltype_dict[table_name] = inferred_col_to_coltype

        return df_coltype_dict
