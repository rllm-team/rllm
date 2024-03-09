__version__ = '0.0.1'

from . import random, tools
from ._tensor_ops import iter_batches
from ._tools import EarlyStopping, Timer

__all__ = [
    # >>> modules
    'random',
    'tools',
    # >>> functions
    'iter_batches',
    # >>> deprecated
    'EarlyStopping',
    'Timer',
]
