from typing import Optional
from . import tools
from ._utils import deprecated


@deprecated('Renamed to `delu.tools.EarlyStopping`')
class EarlyStopping(tools.EarlyStopping):
    """
    <DEPRECATION MESSAGE>
    """

    def forget_bad_updates(self) -> None:
        return self.reset_unsuccessful_updates()


@deprecated('Renamed to `delu.tools.Timer`')
class Timer(tools.Timer):
    """
    <DEPRECATION MESSAGE>
    """

    def __call__(self) -> float:
        return self.elapsed()

    def format(self, format_str: str, /) -> str:
        return self.__format__(format_str)

    def __init__(self, patienc: Optional[int], min_delta: float = 0.0) -> None:
        self._patience = patienc
        self._min_delta = float(min_delta)
        self._best_score: Optional[float] = None
        self._bad_counter = 0

    @property
    def best_score(self) -> Optional[float]:
        """The best score so far.

        If the tracker is just created/reset, return `None`.
        """
        return self._best_score
