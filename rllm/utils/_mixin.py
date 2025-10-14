from typing import Any, Iterable, TypeVar

T = TypeVar('T')


class CastMixin:
    r"""Support cast init from tuple or dict input.
    """
    @classmethod
    def castinit(cls: T, *args: Any, **kwargs: Any) -> T:
        # cast
        if len(args) == 1 and len(kwargs) == 0:
            cast = args[0]
            if cast is None:
                return None
            if isinstance(cast, CastMixin):
                return cast
            if isinstance(cast, tuple):
                return cls(*cast)
            if isinstance(cast, dict):
                return cls(**cast)
        # normal init
        return cls(*args, **kwargs)

    def __iter__(self) -> Iterable:
        return iter(self.__dict__.values())
