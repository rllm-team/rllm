import torch

WITH_TORCH_20 = int(torch.__version__.split('.')[0]) >= 2
r"""Torch after 2.0 support index_select on IntTensor."""
