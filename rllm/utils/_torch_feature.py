import torch

# Torch after 2.0 support index_select on IntTensor.
WITH_TORCH_20 = int(torch.__version__.split(".")[0]) >= 2
