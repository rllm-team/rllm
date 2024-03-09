"""Handy tools for developing deep learning pipelines."""

import datetime
import time
from copy import deepcopy
from typing import Any, Dict, Literal, Optional


class EarlyStopping:
    """Prevents overfitting by stopping training when the validation metric stops improving.

    "Stops improving" means that the best metric value (over the whole training run)
    does not improve for N (``patience``) consecutive epochs.

    **Usage**

    Preventing overfitting by stopping the training
    when the validation metric stops improving:

    >>> def evaluate_model() -> float:
    ...     # Compute and return the metric for the validation set.
    ...     return torch.rand(1).item()
    ...
    >>> # If the validation score does not increase (mode='max')
    >>> # for patience=10 epochs in a row, stop the training.
    >>> early_stopping = delu.EarlyStopping(patience=10, mode='max')
    >>> for epoch in range(1000):
    ...     # Training.
    ...     ...
    ...     # Evaluation
    ...     validation_score = evaluate_model()
    ...     ...
    ...     # Submit the new score.
    ...     early_stopping.update(validation_score)
    ...     # Check whether the training should stop.
    ...     if early_stopping.should_stop():
    ...         break

    Additional technical examples:

    >>> early_stopping = delu.EarlyStopping(2, mode='max')
    >>>
    >>> # Format: (<the best seen score>, <the number of consequtive fails>)
    >>> early_stopping.update(1.0)  # (1.0, 0)
    >>> early_stopping.should_stop()
    False
    >>> early_stopping.update(0.0)  # (1.0, 1)
    >>> early_stopping.should_stop()
    False
    >>> early_stopping.update(2.0)  # (2.0, 0)
    >>> early_stopping.update(1.0)  # (2.0, 1)
    >>> early_stopping.update(2.0)  # (2.0, 2)
    >>> early_stopping.should_stop()
    True

    Resetting the number of the latest consequtive non-improving updates
    without resetting the best seen score:

    >>> early_stopping.reset_unsuccessful_updates()  # (2.0, 0)
    >>> early_stopping.should_stop()
    False
    >>> early_stopping.update(0.0)  # (2.0, 1)
    >>> early_stopping.update(0.0)  # (2.0, 2)
    >>> early_stopping.should_stop()
    True

    The next successfull update resets the number of consequtive fails:

    >>> early_stopping.update(0.0)  # (2.0, 3)
    >>> early_stopping.should_stop()
    True
    >>> early_stopping.update(3.0)  # (3.0, 0)
    >>> early_stopping.should_stop()
    False

    It is possible to completely reset the instance:

    >>> early_stopping.reset()  # (-inf, 0)
    >>> early_stopping.should_stop()
    False
    >>> early_stopping.update(-10.0)   # (-10.0, 0)
    >>> early_stopping.update(-100.0)  # (-10.0, 1)
    >>> early_stopping.update(-10.0)   # (-10.0, 2)
    >>> early_stopping.should_stop()
    True
    """  # noqa: E501

    def __init__(
        self, patience: int, *, mode: Literal['min', 'max'], min_delta: float = 0.0
    ) -> None:
        """
        Args:
            patience: when the number of the latest consequtive non-improving updates
                reaches ``patience``, `EarlyStopping.should_stop` starts returning
                `True` until the next improving update.
            mode: if "min", then the update rule is "the lower value is the better
                value". For "max", it is the opposite.
            min_delta: a new value must differ from the current best value by more
                than ``min_delta`` to be considered as an improvement.
        """
        if patience < 1:
            raise ValueError(
                f'patience must be a positive integer (the provided value: {patience}).'
            )
        if mode not in ('min', 'max'):
            raise ValueError(
                f'mode must be either "min" or "max" (the provided value: "{mode}").'
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
        """Reset the instance completely."""
        self._best_value = None
        self._n_consequtive_bad_updates = 0

    def reset_unsuccessful_updates(self) -> None:
        """Reset the number of the latest consecutive non-improving updates to zero.

        Note that this method does NOT reset the best seen score.
        """
        self._n_consequtive_bad_updates = 0

    def should_stop(self) -> bool:
        """Check whether the early stopping condition is activated.

        See examples in `EarlyStopping`.

        Returns:
            `True` if the number of consequtive bad updates has reached the patience.
            `False` otherwise.
        """
        return self._n_consequtive_bad_updates >= self._patience

    def update(self, value: float) -> None:
        """Submit a new value.

        Args:
            value: the new value.
        """
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
    """A simple **pickle-friendly** timer for measuring execution time.

    `Timer` is applicable to both long-running activies (e.g. a whole program)
    and limited code regions (e.g. training steps).

    - `Timer` can be paused/resumed
      to measure execution time of only relevant activities.
    - `Timer` can be used as a context manager.
    - `Timer` **is pickle-friendly and can be saved to / loaded from a checkpoint.**
    - `Timer` can report the elapsed time as a human-readable string
      with `print`, `str`, `format` and f-strings.

    .. note::

        - Under the hood, `Timer` uses `time.perf_counter` to perform time measurements.
        - `Timer` is not aware of CUDA, so things like `torch.cuda.synchronize` must
          be called explicitly if needed.

    **Usage**

    The common setup for all examples:

    >>> from time import sleep
    >>>
    >>> def train_epoch(): sleep(0.001)
    >>> def evaluate_epoch(): sleep(0.001)

    Measuring the execution time of a training loop:

    >>> # Initially, the timer is not running.
    >>> timer = delu.tools.Timer()
    >>> assert not timer.is_running
    >>>
    >>> # Run the timer.
    >>> timer.run()
    >>> assert timer.is_running
    >>>
    >>> for epoch in range(100):
    ...     train_epoch()
    ...     evaluate_epoch()
    >>> duration = timer.elapsed()

    Same as above, but using a context:

    .. important::

        When entering the context, the timer must not be running.

    >>> # Or simply `with delu.tools.Timer() as timer:`
    >>> timer = delu.tools.Timer()
    >>> with timer:
    ...     # On enter, timer.run() is called.
    ...     assert timer.is_running
    ...     for epoch in range(100):
    ...         train_epoch()
    ...         evaluate_epoch()
    ...
    >>> # On exit, timer.pause() is called.
    >>> assert not timer.is_running
    >>> duration = timer.elapsed()

    Measuring the execution time only of the training activity:

    >>> timer = delu.tools.Timer()
    >>> for epoch in range(100):
    ...     timer.run()       # Start/resume the timer.
    ...     train_epoch()     # Recorded.
    ...     timer.pause()     # Pause the timer.
    ...
    ...     evaluate_epoch()  # Not recorded.
    >>> total_training_duration = timer.elapsed()

    Same as above, but using a context:

    >>> timer = delu.tools.Timer()
    >>> for epoch in range(100):
    ...     with timer:
    ...         train_epoch()
    ...     evaluate_epoch()
    >>> total_training_duration = timer.elapsed()

    Using the timer as a global timer
    (the difference with using `time.perf_counter` directly is that
    the ``start`` variable below can be safely saved to / loaded from a checkpoint):

    >>> timer = delu.tools.Timer()
    >>> timer.run()
    >>> ...  # Other activities.
    >>> start = timer.elapsed()
    >>> sleep(0.001)
    >>> end = timer.elapsed()
    >>> duration = end - start

    Resetting the timer:

    >>> timer.reset()
    >>> assert not timer.is_running
    >>> timer.elapsed() == 0.0
    True

    Reporting the elapsed time in a human-readable format
    (the default format is the same as for `datetime.timedelta`:
    `[days "days" if >0] hours:minutes:seconds[.microseconds if >0]`):

    >>> timer = delu.tools.Timer()
    >>> timer.run()
    >>> sleep(2.0)  # Sleep for two seconds.
    >>> timer.pause()
    >>>
    >>> # print(timer) also works
    >>> str(timer) == format(timer) == f'{timer}'
    >>>
    >>> # Two seconds plus epsilon:
    >>> str(timer).startswith('0:00:02.')
    True
    >>> f'{timer:%Hh %Mm %Ss}'
    '00h 00m 02s'
    >>> format(timer, '%Hh %Mm %Ss')
    '00h 00m 02s'

    A timer can be saved with `torch.save`/`pickle.dump`
    and loaded with `torch.load`/`pickle.load` together with other objects:

    >>> import tempfile
    >>>
    >>> model = nn.Linear(1, 1)
    >>> timer = delu.tools.Timer()
    >>> timer.run()
    >>> sleep(0.001)
    >>> ...
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     path = f'{tmpdir}/checkpoint.pt'
    ...     torch.save(
    ...         {'model': model.state_dict(), 'timer': timer},
    ...         path,
    ...     )
    ...     ...
    ...     checkpoint = torch.load(path)
    ...     model.load_state_dict(checkpoint['model'])
    ...     # The just loaded timer is on pause,
    ...     # so it must be explicitly resumed.
    ...     timer = checkpoint['timer']
    ...     assert not timer.is_running
    ...     timer.run()

    Additional technical examples:

    >>> # Implementing a pause context.
    >>> from contextlib import ExitStack
    >>>
    >>> timer = delu.tools.Timer()
    >>> timer.run()
    >>> ...
    >>> timer.pause()
    >>> with ExitStack() as stack:
    ...     # Call `timer.run()` on exit.
    ...     stack.callback(timer.run)
    ...     ...  # Some activity which is not recorded by the timer.
    >>> timer.is_running
    True
    """

    # mypy cannot infer types from .reset(), so they must be given here
    _start_time: Optional[float]
    _pause_time: Optional[float]
    _shift: float

    def __init__(self) -> None:
        """
        Args:
        """
        self.reset()

    def reset(self) -> None:
        """Reset the timer completely.

        To start using the instance again after resetting,
        the timer must be explicitly run with `Timer.run`.
        """
        self._start_time = None
        self._pause_time = None
        self._shift = 0.0

    @property
    def is_running(self) -> bool:
        """Check if the timer is running."""
        return self._start_time is not None and self._pause_time is None

    def run(self) -> None:
        """Start/resume the timer.

        If the timer is on pause, the method resumes the timer.
        If the timer is running, the method does nothing.
        """
        if self._start_time is None:
            self._start_time = time.perf_counter()
        elif self._pause_time is not None:
            self._shift -= time.perf_counter() - self._pause_time
            self._pause_time = None

    def pause(self) -> None:
        """Pause the timer.

        If the timer is running, the method pauses the timer.
        If the timer was never ``.run()`` or is already on pause,
        the method does nothing.
        """
        if self._start_time is not None:
            if self._pause_time is None:
                self._pause_time = time.perf_counter()

    def elapsed(self) -> float:
        """Get the time elapsed.

        The elapsed time is the time (in seconds) passed since the first ``.run()``
        call up to the ``.elapsed()`` call, minus pauses.

        Returns:
            The elapsed time.
        """
        if self._start_time is None:
            return self._shift
        now = time.perf_counter() if self._pause_time is None else self._pause_time
        return now - self._start_time + self._shift

    def __format__(self, format_spec: str, /) -> str:
        """Format the time elapsed since the start in a human-readable string.

        This is a shortcut for ``time.strftime(format_str, time.gmtime(self()))``.

        Args:
            format_str: the format string passed to `time.strftime`.
        Returns:
            the filled ``format_str``.

        **Usage**

        >>> # xdoctest: +SKIP
        >>> timer = delu.tools.Timer()
        >>> # Let's say that exactly 3661 seconds have passed.
        >>> assert format(timer, '%Hh %Mm %Ss') == '01h 01m 01s'
        """
        return (
            time.strftime(format_spec, time.gmtime(self.elapsed()))
            if format_spec
            else str(datetime.timedelta(seconds=self.elapsed()))
        )

    def __str__(self) -> str:
        return self.__format__('')

    def __enter__(self) -> 'Timer':
        """Measure time within a context.

        **The timer must not be running when entering the context.**

        - On enter, `Timer.run` is called regardless of the current state.
        - On exit, `Timer.pause` is called regardless of the current state.
        """
        if self.is_running:
            raise RuntimeError(
                'The timer cannot be used as a context manager when it is running.'
                ' See the documentation for the workaround.'
            )
        self.run()
        return self

    def __exit__(self, *args) -> bool:  # type: ignore
        """Leave the context and pause the timer."""
        self.pause()
        return False

    def __getstate__(self) -> Dict[str, Any]:
        state = deepcopy(self.__dict__)
        state['_shift'] = self.elapsed()
        state['_start_time'] = None
        state['_pause_time'] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Load the state.

        A time with just loaded state is not running (basically, it is a freshly
        created timer which stores the elapsed time from the loaded state).
        """
        self.__dict__.update(state)
