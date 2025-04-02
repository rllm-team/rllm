from typing import List
from copy import deepcopy

from rllm.data.graph_data import GraphData
from rllm.transforms.graph_transforms import NodeTransform
from rllm.transforms.graph_transforms.functional import remove_training_classes


class RemoveTrainingClasses(NodeTransform):
    r"""Removes classes from the node-level training set as given by
    `data.train_mask`, *e.g.*, in order to get a zero-shot label scenario.

    Args:
        classes (List[int]): The classes to remove from the training set.
    """

    def __init__(self, classes: List[int]):
        self.classes = classes

    def __call__(self, data):
        return self.forward(deepcopy(data))

    def forward(self, data: GraphData):
        assert hasattr(data, "train_mask")
        data.train_mask = remove_training_classes(
            data.train_mask,
            data.y,
            self.classes,
        )
        return data
