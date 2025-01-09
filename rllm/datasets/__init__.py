from .adult import Adult
from .bank_marketing import BankMarketing
from .churn_modelling import ChurnModelling
from .dblp import DBLP
from .imdb import IMDB
from .planetoid import PlanetoidDataset
from .sjtutables.tml1m import TML1MDataset
from .sjtutables.tlf2k import TLF2KDataset
from .sjtutables.tacm12k import TACM12KDataset
from .tape import TAPEDataset
from .titanic import Titanic
from .tagdataset import TAGDataset

__all__ = [
    "Adult",
    "BankMarketing",
    "ChurnModelling",
    "DBLP",
    "IMDB",
    "PlanetoidDataset",
    "TML1MDataset",
    "TAPEDataset",
    "Titanic",
    "TAGDataset",
    "TLF2KDataset",
    "TACM12KDataset",
]
