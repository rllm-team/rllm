from typing import List, Tuple, Dict, Any, Union, Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np
import torch
from torch import Tensor


class Table:
    """
    Wrap pandas DataFrame as a Table, with Tensor primary key.
    This is convenient for torch DataLoader and sampler.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        col_types: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        self.df = df
        self.col_types = col_types

        if 'pkey' in kwargs:
            assert isinstance(kwargs['pkey'], str), "Primary key col should be `str`"
            self.pkey = kwargs['pkey']
            self.pkey_t = torch.as_tensor(self.df[self.pkey].values)
        else:
            try:
                self.pkey = self.df.columns[0]
                self.pkey_t = torch.as_tensor(self.df[self.pkey].values)
            except IndexError:
                raise ValueError("Primary key col should be provided or \
                                  the table should have at least one column")
            
    def __len__(self):
        return len(self.df)