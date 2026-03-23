from typing import TypeVar, Dict, Any, Optional, List


X = TypeVar('X')
Y = TypeVar('Y')


def remap_keys(
    inputs: Dict[X, Any],
    mapping: Dict[X, Y],
    exclude: Optional[List[X]] = None,
    inplace: bool = False,
) -> Optional[Dict[Y, Any]]:
    r"""Remap the keys of the input dictionary using a mapping.

    Args:
        inputs (Dict[X, Any]): The input dictionary whose keys are to be
            remapped.
        mapping (Dict[X, Y]): A mapping from old keys to new keys.
        exclude (List[X], optional): Keys to leave unchanged even if
            present in :obj:`mapping`. (default: :obj:`None`)
        inplace (bool): If set to :obj:`True`, modifies :obj:`inputs` in
            place and returns :obj:`None`. Otherwise returns a new
            dictionary. (default: :obj:`False`)

    Returns:
        Optional[Dict[Y, Any]]: A new dictionary with remapped keys, or
        :obj:`None` if :obj:`inplace=True`.

    Example:
        >>> inputs = {'a': 1, 'b': 2, 'c': 3}
        >>> mapping = {'a': 'A', 'b': 'B'}
        >>> remap_keys(inputs, mapping)
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
