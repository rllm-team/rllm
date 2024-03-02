"""Deep Learning Utilities for PyTorch users."""
__version__ = '0.0.24.dev0'

from . import cuda, data, hardware, nn, random, tools, utils
from ._stream import Stream
from ._tensor_ops import cat, concat, iter_batches, to
from ._tools import EarlyStopping, ProgressTracker, Timer
from ._utilities import evaluation, improve_reproducibility
from .data import collate

# The order is optimized for pdoc.
__all__ = [
    # >>> modules
    'cuda',
    'nn',
    'random',
    'utils',
    'tools',
    # >>> functions
    'to',
    'cat',
    'iter_batches',
    # >>> deprecated
    'EarlyStopping',
    'Timer',
    'data',
    'hardware',
    'collate',
    'concat',
    'evaluation',
    'improve_reproducibility',
    'ProgressTracker',
    'Stream',
]
