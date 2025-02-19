from typing import List, Tuple, Dict, Any, Union

import pandas as pd
from pandas import DataFrame as df
import numpy as np
import torch
from torch import Tensor

from table import Table

class RelationFrame:
    """
    RelationFrame is tables with relations
    """

    def __init__(
        self,
        tables: List[Table],
        **kwargs,
    ):
        self.tables = tables

        if 'relation' in kwargs:
            self.relation = kwargs['relation']
        else:
            self.relation = self._infer_relation()
    
    def _infer_relation(self):
        pass