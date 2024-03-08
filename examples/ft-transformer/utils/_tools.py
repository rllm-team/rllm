import enum
from typing import Optional

from . import tools
from ._utils import deprecated


@deprecated('Renamed to `delu.tools.EarlyStopping`')
class EarlyStopping(tools.EarlyStopping):
    """
    <DEPRECATION MESSAGE>
    """

    def forget_bad_updates(self) -> None:
        """In the new class, see `delu.tools.EarlyStopping.reset_unsuccessful_updates`."""  # noqa: E501
        return self.reset_unsuccessful_updates()


@deprecated('Renamed to `delu.tools.Timer`')
class Timer(tools.Timer):
    """
    <DEPRECATION MESSAGE>
    """

    def __call__(self) -> float:
        """In the new class, see `delu.tools.Timer.elapsed`."""
        return self.elapsed()

    def format(self, format_str: str, /) -> str:
        """In the new class, see the tutorial in `delu.tools.Timer`."""
        return self.__format__(format_str)


class _ProgressStatus(enum.Enum):
    NEUTRAL = enum.auto()
    SUCCESS = enum.auto()
    FAIL = enum.auto()


@deprecated('Instead, use `delu.EarlyStopping` and manually track the best score.')
class ProgressTracker:
    """Helps with early stopping and tracks the best metric value.

    <DEPRECATION MESSAGE>

    For `~ProgressTracker`, **the greater score is the better score**.
    At any moment the tracker is in one of the following states:

    - *success*: the last update increased the best score
    - *fail*: last ``n > patience`` updates did not improve the best score
    - *neutral*: if neither success nor fail

    .. rubric:: Tutorial

    .. testcode::

        progress = delu.ProgressTracker(2)
        progress.update(-999999999)
        assert progress.success  # the first update always updates the best score

        progress.update(123)
        assert progress.success
        assert progress.best_score == 123

        progress.update(0)
        assert not progress.success and not progress.fail

        progress.update(123)
        assert not progress.success and not progress.fail
        progress.update(123)
        # patience is 2 and the best score is not updated for more than 2 steps
        assert progress.fail
        assert progress.best_score == 123  # fail doesn't affect the best score
        progress.update(123)
        assert progress.fail  # still no improvements

        progress.forget_bad_updates()
        assert not progress.fail and not progress.success
        assert progress.best_score == 123
        progress.update(0)
        assert not progress.fail  # just 1 bad update (the patience is 2)

        progress.reset()
        assert not progress.fail and not progress.success
        assert progress.best_score is None
    """

    def __init__(self, patience: Optional[int], min_delta: float = 0.0) -> None:
        """
        Args:
            patience: Allowed number of unsuccessfull updates. For example, if patience
                is 2, then 2 unsuccessfull updates in a row is not a fail,
                but 3 unsuccessfull updates in a row is a fail.
                `None` means "infinite patience" and the progress tracker is never
                in the "fail" state.
            min_delta: the minimal improvement over the current best score
                to count it as success.

        Examples:
            .. testcode::

                progress = delu.ProgressTracker(2)
                progress = delu.ProgressTracker(3, 0.1)
        """
        self._patience = patience
        self._min_delta = float(min_delta)
        self._best_score: Optional[float] = None
        self._status = _ProgressStatus.NEUTRAL
        self._bad_counter = 0

    @property
    def best_score(self) -> Optional[float]:
        """The best score so far.

        If the tracker is just created/reset, return `None`.
        """
        return self._best_score

    @property
    def success(self) -> bool:
        """Check if the tracker is in the "success" state."""
        return self._status == _ProgressStatus.SUCCESS

    @property
    def fail(self) -> bool:
        """Check if the tracker is in the "fail" state."""
        return self._status == _ProgressStatus.FAIL

    def _set_success(self, score: float) -> None:
        self._best_score = score
        self._status = _ProgressStatus.SUCCESS
        self._bad_counter = 0

    def update(self, score: float) -> None:
        """Submit a new score and update the tracker's state accordingly.

        Args:
            score: the score to use for the update.
        """
        if self._best_score is None:
            self._set_success(score)
        elif score > self._best_score + self._min_delta:
            self._set_success(score)
        else:
            self._bad_counter += 1
            self._status = (
                _ProgressStatus.FAIL
                if self._patience is not None and self._bad_counter > self._patience
                else _ProgressStatus.NEUTRAL
            )

    def forget_bad_updates(self) -> None:
        """Reset unsuccessfull update counter and set the status to "neutral".

        Note that this method does NOT reset the best score.
        """
        self._bad_counter = 0
        self._status = _ProgressStatus.NEUTRAL

    def reset(self) -> None:
        """Reset everything."""
        self.forget_bad_updates()
        self._best_score = None
