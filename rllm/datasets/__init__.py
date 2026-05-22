from importlib import import_module
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from rllm.datasets.adult import Adult
    from rllm.datasets.bank_marketing import BankMarketing
    from rllm.datasets.churn_modelling import ChurnModelling
    from rllm.datasets.dblp import DBLP
    from rllm.datasets.imdb import IMDB
    from rllm.datasets.jannis import Jannis
    from rllm.datasets.planetoid import PlanetoidDataset
    from rllm.datasets.sjtutables.tml1m import TML1MDataset
    from rllm.datasets.sjtutables.tlf2k import TLF2KDataset
    from rllm.datasets.sjtutables.tacm12k import TACM12KDataset
    from rllm.datasets.tape import TAPEDataset
    from rllm.datasets.titanic import Titanic
    from rllm.datasets.tagdataset import TAGDataset
    from rllm.datasets.relbench.base import (
        RelBenchDataset,
        RelBenchTask,
        RelBenchTaskType,
        RelBenchTableMeta,
    )
    from rllm.datasets.relbench.f1 import RelF1Dataset
    from rllm.datasets.lakemlb.mstraffic import MSTrafficDataset
    from rllm.datasets.lakemlb.ncbuilding import NCBuildingDataset
    from rllm.datasets.lakemlb.gacars import GACarsDataset
    from rllm.datasets.lakemlb.nnstocks import NNStocksDataset
    from rllm.datasets.lakemlb.lhstocks import LHStocksDataset
    from rllm.datasets.lakemlb.dsmusic import DSMusicDataset

_LAZY_MODULES = {
    "rllm.datasets.adult": (
        "Adult",
    ),
    "rllm.datasets.bank_marketing": (
        "BankMarketing",
    ),
    "rllm.datasets.churn_modelling": (
        "ChurnModelling",
    ),
    "rllm.datasets.dblp": (
        "DBLP",
    ),
    "rllm.datasets.imdb": (
        "IMDB",
    ),
    "rllm.datasets.jannis": (
        "Jannis",
    ),
    "rllm.datasets.planetoid": (
        "PlanetoidDataset",
    ),
    "rllm.datasets.sjtutables.tml1m": (
        "TML1MDataset",
    ),
    "rllm.datasets.sjtutables.tlf2k": (
        "TLF2KDataset",
    ),
    "rllm.datasets.sjtutables.tacm12k": (
        "TACM12KDataset",
    ),
    "rllm.datasets.tape": (
        "TAPEDataset",
    ),
    "rllm.datasets.titanic": (
        "Titanic",
    ),
    "rllm.datasets.tagdataset": (
        "TAGDataset",
    ),
    "rllm.datasets.relbench.base": (
        "RelBenchDataset",
        "RelBenchTask",
        "RelBenchTaskType",
        "RelBenchTableMeta",
    ),
    "rllm.datasets.relbench.f1": (
        "RelF1Dataset",
    ),
    "rllm.datasets.lakemlb.mstraffic": (
        "MSTrafficDataset",
    ),
    "rllm.datasets.lakemlb.ncbuilding": (
        "NCBuildingDataset",
    ),
    "rllm.datasets.lakemlb.gacars": (
        "GACarsDataset",
    ),
    "rllm.datasets.lakemlb.nnstocks": (
        "NNStocksDataset",
    ),
    "rllm.datasets.lakemlb.lhstocks": (
        "LHStocksDataset",
    ),
    "rllm.datasets.lakemlb.dsmusic": (
        "DSMusicDataset",
    ),
}

__all__ = [
    name
    for names in _LAZY_MODULES.values()
    for name in names
]

_LAZY_ATTRS = {
    name: module_name
    for module_name, names in _LAZY_MODULES.items()
    for name in names
}


def __getattr__(name):
    if name in _LAZY_ATTRS:
        module = import_module(_LAZY_ATTRS[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
