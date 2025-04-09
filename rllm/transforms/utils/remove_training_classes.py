from typing import List

from rllm.data.graph_data import GraphData
from rllm.transforms.utils import BaseTransform
from rllm.transforms.utils.functional import remove_training_classes


class RemoveTrainingClasses(BaseTransform):
    r"""Removes classes from the node-level training set as given by
    `data.train_mask`, *e.g.*, in order to get a zero-shot label scenario.

    Args:
        classes (List[int]): The classes to remove from the training set.
    """

    def __init__(self, classes: List[int]):
        self.classes = classes

    def forward(self, data: GraphData):
        assert hasattr(data, "train_mask")
        data.train_mask = remove_training_classes(
            data.train_mask,
            data.y,
            self.classes,
        )
        return data
