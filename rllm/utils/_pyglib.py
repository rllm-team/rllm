"""
Wrapper for pyglib interface.
"""

import warnings
import importlib

try:
    importlib.import_module("pyg_lib")
    WITH_PYG_LIB = True

    # Check torch ops registered
    import torch

    assert hasattr(
        torch.ops.pyg, "hetero_neighbor_sample"
    ), "pyg_lib is installed, but torch ops are not registered."
except ImportError:
    WITH_PYG_LIB = False
    warnings.warn("pyg_lib is not installed.")
