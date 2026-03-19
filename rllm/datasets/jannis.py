import os
import os.path as osp
from typing import Optional

import pandas as pd

# import arff
from scipy.io import arff

from rllm.types import ColType
from rllm.data.table_data import TableData
from rllm.datasets.dataset import Dataset
from rllm.utils.download import download_url


class Jannis(Dataset):
    r"""The `Jannis dataset <https://www.openml.org/d/41168>`__
    is a subset of the SAIAPR TC-12 collection,
    which was introduced during the `2015/2016 AutoML Challenge
    <https://link.springer.com/chapter/10.1007/978-3-030-05318-5_10>`__.
    The SAIAPR TC-12 dataset, prepared by Hugo Jair Escalante and
    Michael Grubinger, originates from the IAPR TC-12 benchmark
    (http://imageclef.org/SIAPRdata). It extends the original collection
    through fine-grained manual segmentation of 20,000 images,
    resulting in a total of 99,535 segmented regions. The dataset is widely
    used in machine learning for multi-label classification tasks, where
    the goal is to predict multiple semantic labels for each image region
    based on its visual and spatial features.  It presents significant
    challenges due to the high dimensionality of features, class imbalance,
    and the complex hierarchical relationships among labels.

    Each region includes not only the segmentation mask and the corresponding cropped
    image, but also a set of visual features and manually assigned semantic labels
    derived from a predefined concept hierarchy. Additionally, spatial relationships
    among regions (such as adjacency, separation, and vertical alignment) are provided.
    The visual features encompass geometric attributes (e.g., area, width,
    height, convexity) as well as color statistics (mean, standard deviation,
    and skewness) computed in both RGB and CIE-Lab color spaces. This dataset
    serves as a rich and structured resource for research in image understanding,
    region-level annotation, and content-based image retrieval.

    Args:
        cached_dir (str): Root directory where dataset should be saved.
        forced_reload (bool): If set to `True`, this dataset will be
            re-processed again.

    .. parsed-literal::

        Statics:
        Name    Region  Features
        Size    83,733  54


    """

    url = "https://www.openml.org/data/download/19335691/file1c558ee247d.arff"

    def __init__(self, cached_dir: str, forced_reload: Optional[bool] = False) -> None:
        self.name = "jannis"
        root = os.path.join(cached_dir, self.name)
        super().__init__(root, force_reload=forced_reload)
        self.data_list = [TableData.load(self.processed_paths[0])]

    @property
    def raw_filenames(self):
        return ["jannis.arff"]

    @property
    def processed_filenames(self):
        return ["data.pt"]

    def process(self):
        r"""
        process data and save to './cached_dir/{dataset}/processed/'.
        """
        os.makedirs(self.processed_dir, exist_ok=True)
        path = osp.join(self.raw_dir, self.raw_filenames[0])
        with open(path, "r") as f:
            data, meta = arff.loadarff(f)

        # Note: the order of column in col_types must
        # correspond to the order of column in files,
        # except target column.
        col_types = {}
        colunms = meta.names()
        df = pd.DataFrame(data)

        for name, type in zip(meta.names(), meta.types()):
            if type == "numeric":
                col_types[name] = ColType.NUMERICAL
            else:
                col_types[name] = ColType.CATEGORICAL
        df = pd.DataFrame(data, columns=colunms)

        data = TableData(
            df=df,
            col_types=col_types,
            target_col="class",
        )

        data.save(self.processed_paths[0])

    def download(self):
        os.makedirs(self.raw_dir, exist_ok=True)
        download_url(self.url, self.raw_dir, self.raw_filenames[0])

    def __len__(self):
        return 1

    def __getitem__(self, index: int):
        if index != 0:
            raise IndexError
        return self.data_list[index]
