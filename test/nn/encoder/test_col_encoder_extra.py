import pytest
import torch

from rllm.nn.encoder.col_encoder._cyclic_encoder import CyclicEncoder
from rllm.nn.encoder.col_encoder._positional_encoder import PositionalEncoder
from rllm.nn.encoder.col_encoder._reshape_encoder import ReshapeEncoder
from rllm.nn.encoder.col_encoder._textembedding_encoder import TextEmbeddingEncoder
from rllm.nn.encoder.col_encoder._timestamp_encoder import TimestampEncoder
from rllm.nn.encoder.col_encoder._transtab_num_embedding_encoder import (
    TransTabNumEmbeddingEncoder,
)
from rllm.nn.encoder.col_encoder._transtab_word_embedding_encoder import (
    TransTabWordEmbeddingEncoder,
)
from rllm.types import StatType


def test_positional_and_cyclic_encoders_shape_and_validation():
    x = torch.tensor([[0.0, 1.0], [2.0, 3.0]])

    assert PositionalEncoder(out_size=4)(x).shape == (2, 2, 4)
    assert CyclicEncoder(out_size=4)(torch.tensor([[0.0, 0.5, 1.0]])).shape == (1, 3, 4)

    with pytest.raises(ValueError):
        PositionalEncoder(out_size=3)
    with pytest.raises(ValueError):
        CyclicEncoder(out_size=3)
    with pytest.raises(ValueError):
        PositionalEncoder(out_size=4)(torch.tensor([-1.0]))
    with pytest.raises(ValueError):
        CyclicEncoder(out_size=4)(torch.tensor([1.5]))


def test_reshape_encoder_allows_default_optional_stats_list():
    encoder = ReshapeEncoder()
    encoder.post_init()

    out = encoder(torch.randn(3, 2))

    assert out.shape == (3, 2, 1)


def test_text_embedding_encoder_handles_each_column_independently():
    encoder = TextEmbeddingEncoder(
        out_dim=5,
        stats_list=[{StatType.EMB_DIM: 3}, {StatType.EMB_DIM: 3}],
    )
    encoder.post_init()
    feat = torch.randn(4, 2, 3)

    out = encoder(feat)

    assert out.shape == (4, 2, 5)
    changed = feat.clone()
    changed[:, 0, :] += 10
    changed_out = encoder(changed)
    assert torch.allclose(changed_out[:, 1, :], out[:, 1, :])


def test_timestamp_encoder_fills_nan_and_outputs_requested_dimension():
    encoder = TimestampEncoder(
        out_dim=6,
        out_size=4,
        fill_nan=True,
        stats_list=[
            {
                StatType.YEAR_RANGE: [2020, 2022],
                StatType.MEDIAN_TIME: "2021-02-03 04:05:06",
            }
        ],
    )
    encoder.post_init()
    feat = torch.tensor(
        [
            [[2020, 0, 0, 0, 0, 0, 0]],
            [[-1, -1, -1, -1, -1, -1, -1]],
        ],
        dtype=torch.long,
    )

    out = encoder(feat)

    assert out.shape == (2, 1, 6)
    assert torch.isfinite(out).all()


def test_transtab_word_and_num_embedding_encoders():
    word = TransTabWordEmbeddingEncoder(vocab_size=20, out_dim=6, padding_idx=0)
    token_ids = torch.tensor([[1, 2, 0], [3, 4, 5]])

    assert word(token_ids).shape == (2, 3, 6)

    num = TransTabNumEmbeddingEncoder(hidden_dim=6)
    col_emb = torch.randn(3, 6)
    raw_vals = torch.tensor([[1.0, 2.0, 3.0], [0.0, -1.0, 4.0]])

    assert num(col_emb, raw_vals).shape == (2, 3, 6)
