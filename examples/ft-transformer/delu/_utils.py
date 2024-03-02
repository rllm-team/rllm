# This module contains internal tools imported by other modules.

import functools
import inspect
import warnings

from .exceptions import DeLUDeprecationWarning


def deprecated(message: str):
    warning_sign = '**[DEPRECATED]**'

    def decorator(item):
        assert item.__doc__ is not None
        assert '<DEPRECATION MESSAGE>' in item.__doc__
        docstring = item.__doc__.replace(
            '<DEPRECATION MESSAGE>',
            f'{warning_sign} ({message})' if message else warning_sign,
        )

        def warn(item_type):
            warnings.warn(
                f'The {item_type} {item.__qualname__}` is deprecated'
                ' and will be removed in future releases. ' + message,
                DeLUDeprecationWarning,
            )

        if isinstance(item, type):
            wrapper = item
            wrapper.__doc__ = docstring
        else:
            assert inspect.isfunction(item)

            @functools.wraps(item)
            def wrapper(*args, **kwargs):
                warn('function')
                return item(*args, **kwargs)

            wrapper.__doc__ = docstring

        return wrapper

    return decorator
