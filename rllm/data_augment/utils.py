from __future__ import annotations

import numpy as np


def infer_random_state(
    random_state: int | np.random.RandomState | np.random.Generator | None,
) -> tuple[int, np.random.Generator]:
    """Infer a stable seed and NumPy Generator from common random-state inputs."""
    if isinstance(random_state, (int, np.integer)):
        np_rng = np.random.default_rng(random_state)
        static_seed = int(random_state)
    elif isinstance(random_state, np.random.RandomState):
        static_seed = int(random_state.randint(0, 2**31))
        np_rng = np.random.default_rng(static_seed)
    elif isinstance(random_state, np.random.Generator):
        np_rng = random_state
        static_seed = int(np_rng.integers(0, 2**31))
    elif random_state is None:
        np_rng = np.random.default_rng()
        static_seed = int(np_rng.integers(0, 2**31))
    else:
        raise ValueError(f"Invalid random_state {random_state}")

    return static_seed, np_rng
