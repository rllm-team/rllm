import random
import secrets
from typing import Optional

import numpy as np
import torch
import torch.cuda

__all__ = ['seed']

_2_pow_64 = 1 << 64


def seed(base_seed: Optional[int], /, *, one_cuda_seed: bool = False) -> int:
    if base_seed is None:
        base_seed = secrets.randbelow(_2_pow_64)
    if not (0 <= base_seed < _2_pow_64):
        raise ValueError(
            'base_seed must be a non-negative integer from [0, 2**64).'
            f' The provided value: {base_seed=}'
        )

    sequence = np.random.SeedSequence(base_seed)

    def generate_state(*args, **kwargs) -> np.ndarray:
        new_sequence = sequence.spawn(1)[0]
        return new_sequence.generate_state(*args, **kwargs)

    # To generate a 128-bit seed for the standard library,
    # two uint64 numbers are generated and concatenated (literally).
    state_std = generate_state(2, dtype=np.uint64).tolist()
    random.seed(state_std[0] * _2_pow_64 + state_std[1])
    del state_std

    np.random.seed(generate_state(4))

    torch.manual_seed(int(generate_state(1, dtype=np.uint64)[0]))

    if not torch.cuda._is_in_bad_fork():
        if one_cuda_seed:
            torch.cuda.manual_seed_all(
                int(generate_state(1, dtype=np.uint64)[0]))
        else:
            if torch.cuda.is_available():
                torch.cuda.init()
                for i in range(torch.cuda.device_count()):
                    torch.cuda.default_generators[i].manual_seed(
                        int(generate_state(1, dtype=np.uint64)[0])
                    )

    return base_seed
