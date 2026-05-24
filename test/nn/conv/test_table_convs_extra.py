import pytest
import torch

from rllm.nn.conv.table_conv import ExcelFormerConv, ResNetConv, SAINTConv, TransTabConv, TromptConv
from rllm.nn.conv.table_conv.excelformer_conv import GLULayer, SemiPermeableAttention
from rllm.nn.conv.table_conv.transtab_conv import _get_activation_fn
from rllm.types import ColType, StatType


def test_excelformer_components_and_conv_shape():
    x = torch.randn(4, 5, 8)

    assert GLULayer(8, 6)(x).shape == (4, 5, 6)

    attn = SemiPermeableAttention(dim=8, num_heads=2, head_dim=4, dropout=0.0)
    mask = attn.get_attention_mask((4, 2, 5, 5), x.device)

    assert mask.shape == (4, 2, 5, 5)
    assert mask[0, 0, 0, 1].item() < -1000
    assert mask[0, 0, 2, 1].item() == 0
    assert attn(x).shape == x.shape
    assert ExcelFormerConv(conv_dim=8, num_heads=2, head_dim=4, dropout=0.0)(x).shape == x.shape


def test_resnet_conv_normalization_and_shortcut_shapes():
    x = torch.randn(6, 4)

    assert ResNetConv(4, 4, normalization=None, dropout=0.0)(x).shape == (6, 4)
    assert ResNetConv(4, 7, normalization="layer_norm", dropout=0.0)(x).shape == (6, 7)
    assert ResNetConv(4, 7, normalization="batch_norm", dropout=0.0)(x).shape == (6, 7)


def test_saint_conv_column_and_row_attention_shape():
    x = torch.randn(4, 3, 8)
    conv = SAINTConv(conv_dim=8, num_cols=3, num_heads=2, dropout=0.0)

    assert conv(x).shape == x.shape


def test_transtab_conv_masks_norm_modes_and_activation_validation():
    x = torch.randn(3, 4, 8)
    valid_mask = torch.ones(3, 4, dtype=torch.bool)
    valid_mask[0, -1] = False

    conv = TransTabConv(conv_dim=8, nhead=2, dim_feedforward=16, dropout=0.0, activation="gelu")
    assert conv(x, src_key_padding_mask=valid_mask).shape == x.shape

    no_norm = TransTabConv(conv_dim=8, nhead=2, dim_feedforward=16, dropout=0.0, use_layer_norm=False)
    assert no_norm(x, src_key_padding_mask=valid_mask).shape == x.shape

    norm_first = TransTabConv(conv_dim=8, nhead=2, dim_feedforward=16, dropout=0.0, norm_first=True)
    assert norm_first(x, src_key_padding_mask=valid_mask).shape == x.shape

    assert _get_activation_fn("relu")(torch.tensor([-1.0, 1.0])).tolist() == [0.0, 1.0]
    with pytest.raises(RuntimeError):
        _get_activation_fn("bad")


def test_trompt_conv_with_raw_feature_dict():
    metadata = {
        ColType.NUMERICAL: [
            {StatType.MEAN: 0.0, StatType.STD: 1.0},
            {StatType.MEAN: 1.0, StatType.STD: 2.0},
        ],
        ColType.CATEGORICAL: [
            {StatType.COUNT: 4},
        ],
    }
    conv = TromptConv(in_dim=3, out_dim=6, num_prompts=2, metadata=metadata, num_groups=1)
    feat_dict = {
        ColType.NUMERICAL: torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
        ColType.CATEGORICAL: torch.tensor([[0], [2]]),
    }
    x_prompt = torch.randn(2, 2, 6)

    out = conv(feat_dict, x_prompt)

    assert out.shape == (2, 2, 6)
