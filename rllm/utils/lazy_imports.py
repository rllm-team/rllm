from importlib import import_module
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple


def define_lazy_imports(
    module_name: str,
    module_globals: Dict[str, Any],
    lazy_modules: Mapping[str, Sequence[str]],
    lazy_submodules: Optional[Mapping[str, str]] = None,
) -> Tuple[List[str], Callable[[str], Any], Callable[[], List[str]]]:
    all_names = [
        name
        for names in lazy_modules.values()
        for name in names
    ]
    lazy_attrs = {
        name: import_path
        for import_path, names in lazy_modules.items()
        for name in names
    }
    submodules = dict(lazy_submodules or {})
    prefix = f"{module_name}."
    for import_path in lazy_modules:
        if not import_path.startswith(prefix):
            continue
        submodule_name = import_path[len(prefix):].split(".", 1)[0]
        submodules.setdefault(submodule_name, f"{module_name}.{submodule_name}")

    def __getattr__(name: str) -> Any:
        if name in lazy_attrs:
            module = import_module(lazy_attrs[name])
            value = getattr(module, name)
            module_globals[name] = value
            return value
        if name in submodules:
            module = import_module(submodules[name])
            module_globals[name] = module
            return module
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")

    def __dir__() -> List[str]:
        return sorted(set(module_globals) | set(all_names) | set(submodules))

    return all_names, __getattr__, __dir__
