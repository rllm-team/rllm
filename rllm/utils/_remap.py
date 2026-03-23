from typing import TypeVar, Dict, Any, Optional, List, Union


X = TypeVar("X")
Y = TypeVar("Y")


def remap_keys(
    inputs: Dict[X, Any],
    mapping: Dict[X, Y],
    exclude: Optional[List[X]] = None,
) -> Dict[Union[X, Y], Any]:
    r"""Remap the keys of the input dictionary using a mapping.

    Args:
        inputs (Dict[X, Any]): The input dictionary whose keys are to be
            remapped.
        mapping (Dict[X, Y]): A mapping from old keys to new keys.
        exclude (List[X], optional): Keys to leave unchanged even if
            present in :obj:`mapping`. (default: :obj:`None`)

    Returns:
        Dict[Union[X, Y], Any]: A new dictionary with remapped keys.

    Example:
        >>> inputs = {'a': 1, 'b': 2, 'c': 3}
        >>> mapping = {'a': 'A', 'b': 'B'}
        >>> remap_keys(inputs, mapping)
        {'A': 1, 'B': 2, 'c': 3}
    """
    exclude = exclude or []
    return {k if k in exclude else mapping.get(k, k): v for k, v in inputs.items()}
