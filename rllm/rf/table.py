"""
Deprecated. Use rllm.data.table_data.TableData instead.
"""
from typing import List
import warnings

import pandas as pd


class Table:
    """
    Wrap pandas DataFrame as a Table, with Tensor primary key.
    This is convenient for torch DataLoader and sampler.

    TODO: for now, ignore **union** primary or foreign keys.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        **kwargs,
    ):
        self.df = df

        if 'pkey' in kwargs:
            if isinstance(kwargs['pkey'], str) and kwargs['pkey'] in self.df.columns:
                self.pkey = (kwargs['pkey'],)
            elif isinstance(kwargs['pkey'], tuple) and all(pkey in self.df.columns for pkey in kwargs['pkey']):
                self.pkey = kwargs['pkey']
        else:
            try:
                self.pkey = (self.df.columns[0],)
            except IndexError:
                raise ValueError("Primary key col should be provided or \
                                  the table should have at least one column.")

        if 'fkeys' in kwargs:
            self._fkeys = kwargs['fkeys']
        else:
            self._fkeys = None

    def __len__(self):
        return self.df.shape[0]

    @property
    def fkeys(self):
        return self._fkeys

    @fkeys.setter
    def fkeys(self, fkeys: List[str]):
        assert all(fkey in self.df.columns for fkey in fkeys), "Foreign keys should be in the table."
        if self._fkeys:
            warnings.warn("Foreign keys are being overwritten.")
        self._fkeys = fkeys
