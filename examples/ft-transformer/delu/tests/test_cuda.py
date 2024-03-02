import gc
from unittest.mock import patch

import pytest
import torch

import delu


@pytest.mark.parametrize('gpu', [False, True])
def test_free_memory(gpu):
    with patch('gc.collect') as _, patch('torch.cuda.empty_cache') as _, patch(
        'torch.cuda.synchronize'
    ) as _, patch('torch.cuda.is_available', lambda: gpu) as _:
        delu.cuda.free_memory()
        gc.collect.assert_called_once()
        if gpu:
            torch.cuda.synchronize.assert_called_once()
            torch.cuda.empty_cache.assert_called_once()
