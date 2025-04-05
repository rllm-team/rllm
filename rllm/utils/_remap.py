from typing import TypeVar, Dict, Any, Optional, List


X = TypeVar('X')
Y = TypeVar('Y')


def remap_keys(
    inputs: Dict[X, Any],
    mapping: Dict[X, Y],
    exclude: Optional[List[X]] = None,
    inplace: bool = False,
) -> Optional[Dict[Y, Any]]:
    r"""Remap the keys in the input dictionary.

    Args:
        inputs (Dict[X, Any]): The input dictionary.
        mapping (Dict[X, Y]): The mapping dictionary.
        exclude (List[X], optional): The keys to exclude.
            (default: `None`)
        inplace (bool, optional): If set to `True`, will modify the input dictionary
            in place. Otherwise, will return a new dictionary.
            (default: `False`)

    Example:
        >>> inputs = {'a': 1, 'b': 2, 'c': 3}
        >>> mapping = {'a': 'A', 'b': 'B'}
        >>> remap(inputs, mapping)
        {'A': 1, 'B': 2, 'c': 3}
    """
    if exclude is None:
        exclude = []
    out = inputs if inplace else inputs.copy()
    for key, value in inputs.items():
        if key in exclude:
            continue
        if key in mapping:
            out[mapping[key]] = value
            if not inplace:
                del out[key]
    return out if not inplace else None
