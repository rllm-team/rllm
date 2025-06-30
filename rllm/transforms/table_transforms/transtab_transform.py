from __future__ import annotations
from typing import Any, Dict, List
import torch
from torch import nn, Tensor
from rllm.transforms.table_transforms import TableTransform
from rllm.types import ColType
from rllm.transforms.table_transforms import ColNormalize
import math
import collections

# class TransTabTransform(TableTransform):
'''
The process of TransTab is 
"extract ID/original value → send to trainable embedding/linear layer → get learnable feature representation → concatenate".
There is no "static vector" part that is repeated once/one-hot and then directly thrown into the subsequent model.
Therefore, no transform is needed.
'''
