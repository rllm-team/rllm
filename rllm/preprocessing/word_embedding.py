from dataclasses import dataclass
from typing import Callable, Optional

from pandas import Series
import torch
from torch import Tensor
from tqdm import tqdm


@dataclass
class TextEmbedderConfig:
    text_embedder: Callable[[list[str]], Tensor]
    batch_size: Optional[int] = None


def embed_text_column(
    col_series: Series,
    config: TextEmbedderConfig,
) -> Tensor:
    """
    Embed a pandas Series of texts using the provided embedder.
    Returns a float Tensor of shape [N, D].
    """
    embedder = config.text_embedder
    batch_size = config.batch_size
    assert embedder is not None, "Need an embedder for text column!"

    col_str = col_series.astype(str)
    col_list = col_str.to_list()

    if batch_size is None:
        embeddings = embedder(col_list)
    else:
        emb_list: list[Tensor] = []
        for i in tqdm(
            range(0, len(col_list), batch_size), desc="Embedding raw data in mini-batch"
        ):
            emb = embedder(col_list[i : i + batch_size])
            emb_list.append(emb)
        embeddings = torch.cat(emb_list, dim=0)
    return embeddings.float()
