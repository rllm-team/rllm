from typing import List
from torch import Tensor


def remove_training_classes(mask: Tensor, labels: Tensor, classes: List[int]):
    r"""Removes classes from the node-level training set as given by
    `mask`, *e.g.*, in order to get a zero-shot label scenario.

    Args:
        mask (Tensor): The train mask.
        labels (Tensor): The label for each node.
        classes (List[int]): The classes to remove from the training set.
    """
    mask = mask.clone()
    for i in classes:
        mask[labels == i] = False
    return mask
