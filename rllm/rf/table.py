"""
Deprecated. Use rllm.data.table_data.TableData instead.
"""
from typing import List, Tuple, Dict, Any, Union, Optional
from dataclasses import dataclass
from enum import Enum
import warnings

import pandas as pd
import numpy as np
import torch
from torch import Tensor


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



if __name__ == "__main__":
    # Test Table
    df = pd.DataFrame({
        "UserID": [1, 1, 1, 1, 1],
        "MovieID": [1177, 656, 903, 3340, 2287],
        "Rating": [5, 3, 3, 4, 5],
        "Timestamp": [978300760, 978302109, 978301968, 978300275, 978824291]
    })
    table = Table(df, pkey=('UserID', 'MovieID'), fkeys=['UserID', 'MovieID'])
    print(table.pkey)
    print(len(table))
    print(table.df)
    print(table.fkeys)
    # table.fkeys = ['a']
    # print(table.fkeys)
