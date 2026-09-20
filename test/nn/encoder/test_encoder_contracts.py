from __future__ import annotations

import torch

from rllm.nn.encoder import FTTransformerPreEncoder
from rllm.types import ColType, StatType


def test_ft_preencoder_uses_feature_dict_order_and_handles_missing_categories():
    metadata = {
        ColType.NUMERICAL: [{StatType.MEAN: 0.0, StatType.STD: 1.0}],
        ColType.CATEGORICAL: [{StatType.COUNT: 2}],
    }
    encoder = FTTransformerPreEncoder(out_dim=2, metadata=metadata)
    numeric_encoder = encoder.col_encoder_dict[ColType.NUMERICAL.value]
    category_encoder = encoder.col_encoder_dict[ColType.CATEGORICAL.value]
    with torch.no_grad():
        numeric_encoder.weight.fill_(1.0)
        numeric_encoder.bias.zero_()
        category_encoder.emb.weight.copy_(
            torch.tensor([[0.0, 0.0], [10.0, 11.0], [20.0, 21.0]])
        )

    features = {
        ColType.CATEGORICAL: torch.tensor([[-1], [1]]),
        ColType.NUMERICAL: torch.tensor([[2.0], [4.0]]),
    }
    encoded = encoder(features)
    by_type = encoder(features, return_dict=True)

    assert encoded.shape == (2, 2, 2)
    assert torch.equal(encoded[:, 0], torch.tensor([[0.0, 0.0], [20.0, 21.0]]))
    assert torch.allclose(encoded[:, 1], torch.tensor([[2.0, 2.0], [4.0, 4.0]]), atol=1e-5)
    assert list(by_type) == [ColType.CATEGORICAL, ColType.NUMERICAL]
