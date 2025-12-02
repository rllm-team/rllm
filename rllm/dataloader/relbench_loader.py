from typing import Optional, List, Tuple, Callable

import torch
from torch import Tensor

from rllm.data import GraphData
from rllm.datasets import RelBenchDataset, RelBenchTask


class NeighborLoader(torch.utils.data.DataLoader):

    def __init__(
        self,
        dataset: RelBenchDataset
    ):
        dataset.load_all()  # make sure dataset is processed


