from typing import Any, Optional, List

import torch
from torch import Tensor
import torch.utils
import torch.utils.data

class EntryLoader(torch.utils.data.DataLoader):
    
    def __init__(
        self,
        rf: Any,
        batch_size: int = 1,
    ):
        pass