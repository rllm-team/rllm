import torch

mask = torch.tensor([True, False, True, False, True])
print(mask.nonzero())
print(mask.nonzero().flatten())