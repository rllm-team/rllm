import datetime
import time
from copy import deepcopy
from typing import Any, Dict, Literal, Optional


class EarlyStopping:
    def __init__(
        self, patience: int, *, mode: Literal['min', 'max'],
        min_delta: float = 0.0
    ) -> None:
        if patience < 1:
            raise ValueError(
                f'patience must be a provided value: {patience}).'
            )
        if mode not in ('min', 'max'):
            raise ValueError(
                f'mode must be either(the provided value: "{mode}").'
            )
        if min_delta < 0.0:
            raise ValueError(
                'min_delta must be a non-negative number'
                f' (the provided value: {min_delta}).'
            )
        self._patience = patience
        self._maximize = mode == 'max'
        self._min_delta = min_delta
        self._best_value: Optional[float] = None
        self._n_consequtive_bad_updates = 0

    def reset(self) -> None:
        self._best_value = None
        self._n_consequtive_bad_updates = 0

    def reset_unsuccessful_updates(self) -> None:
        self._n_consequtive_bad_updates = 0

    def should_stop(self) -> bool:
        return self._n_consequtive_bad_updates >= self._patience

    def update(self, value: float) -> None:
        success = (
            True
            if self._best_value is None
            else value > self._best_value + self._min_delta
            if self._maximize
            else value < self._best_value - self._min_delta
        )
        if success:
            self._best_value = value
            self._n_consequtive_bad_updates = 0
        else:
            self._n_consequtive_bad_updates += 1


class Timer:
    _start_time: Optional[float]
    _pause_time: Optional[float]
    _shift: float

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._start_time = None
        self._pause_time = None
        self._shift = 0.0

    @property
    def is_running(self) -> bool:
        return self._start_time is not None and self._pause_time is None

    def run(self) -> None:
        if self._start_time is None:
            self._start_time = time.perf_counter()
        elif self._pause_time is not None:
            self._shift -= time.perf_counter() - self._pause_time
            self._pause_time = None

    def pause(self) -> None:
        if self._start_time is not None:
            if self._pause_time is None:
                self._pause_time = time.perf_counter()

    def elapsed(self) -> float:
        if self._start_time is None:
            return self._shift
        if self._pause_time is None:
            n = time.perf_counter()
        else:
            n = self._pause_time
        return n - self._start_time + self._shift

    def __format__(self, format_spec: str, /) -> str:
        return (
            time.strftime(format_spec, time.gmtime(self.elapsed()))
            if format_spec
            else str(datetime.timedelta(seconds=self.elapsed()))
        )

    def __str__(self) -> str:
        return self.__format__('')

    def __enter__(self) -> 'Timer':
        if self.is_running:
            raise RuntimeError(
                'The timer cannot be used as a context manager when it is.'
                ' See the documentation for the workaround.'
            )
        self.run()
        return self

    def __exit__(self, *args) -> bool:  # type: ignore
        self.pause()
        return False

    def __getstate__(self) -> Dict[str, Any]:
        state = deepcopy(self.__dict__)
        state['_shift'] = self.elapsed()
        state['_start_time'] = None
        state['_pause_time'] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
