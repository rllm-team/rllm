from typing import Any, Iterable, TypeVar

T = TypeVar('T')


class CastMixin:
    r"""Support cast init from tuple or dict input.
    """
    @classmethod
    def castinit(cls: T, *args: Any, **kwargs: Any) -> T:
        r"""Construct an instance from positional, tuple, or dict arguments.

        If a single positional argument is given and it is a :class:`tuple`,
        it is unpacked as positional arguments. If it is a :class:`dict`,
        it is unpacked as keyword arguments.

        Returns:
            T: A new instance of the calling class.
        """
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
        r"""Iterate over instance attribute values in definition order."""
        return iter(self.__dict__.values())
