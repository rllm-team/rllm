from typing import TypeVar, Dict, Any, Optional, List, Union


X, Y = TypeVar('X'), TypeVar('Y')


def remap_dict_keys(
    inputs: Dict[X, Any],
    mapping: Dict[X, Y],
    exclude: Optional[List[X]] = None,
) -> Dict[Union[X, Y], Any]:
    """Remap dict keys according to a given mapping."""
    exclude = exclude or []
    return {
        k if k in exclude else mapping.get(k, k): v
        for k, v in inputs.items()
    }
