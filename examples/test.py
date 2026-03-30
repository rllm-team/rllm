from __future__ import annotations

from rllm.types import StatType
from rllm.nn.encoder.col_encoder._remove_empty_features_encoder import (
    RemoveEmptyFeaturesEncoder,
)
from rllm.nn.encoder.col_encoder._nan_handling_encoder import NanHandlingEncoder
from rllm.nn.encoder.col_encoder._input_normalization_encoder import (
    InputNormalizationEncoder,
)
from rllm.nn.encoder.col_encoder._variable_num_features_encoder import (
    VariableNumFeaturesEncoder,
)
from rllm.nn.encoder.col_encoder._linear_encoder import LinearEncoder
from rllm.nn.encoder.col_encoder._frequency_feature_encoder import (
    FrequencyFeatureEncoder,
)


def _print_state_dict(name: str, module) -> None:
    print(f"\n=== {name} ===")
    sd = module.state_dict()
    if len(sd) == 0:
        print("(empty state_dict)")
        return
    for k, v in sd.items():
        print(f"{k}: shape={tuple(v.shape)}, dtype={v.dtype}")


def main() -> None:
    num_features = 5
    stats_list = [
        {
            StatType.MEAN: 0.0,
            StatType.STD: 1.0,
            StatType.MIN: -10.0,
            StatType.MAX: 10.0,
        }
        for _ in range(num_features)
    ]

    # TabPFN 默认链路中常见的 col encoders
    remove_empty = RemoveEmptyFeaturesEncoder(stats_list=stats_list)
    nan_handling = NanHandlingEncoder(stats_list=stats_list)
    input_norm = InputNormalizationEncoder(
        stats_list=stats_list,
        normalize_x=True,
        remove_outliers=False,
    )
    var_num_feat = VariableNumFeaturesEncoder(
        num_features=num_features,
        normalize_by_used_features=True,
        normalize_by_sqrt=True,
        stats_list=stats_list,
    )
    linear = LinearEncoder(
        in_dim=1,
        out_dim=16,
        stats_list=stats_list,
        use_bias=False,
    )

    # 非默认（num_frequencies>0 时）可选链路
    freq = FrequencyFeatureEncoder(
        num_frequencies=2,
        stats_list=stats_list,
    )

    modules = {
        "RemoveEmptyFeaturesEncoder": remove_empty,
        "NanHandlingEncoder": nan_handling,
        "InputNormalizationEncoder": input_norm,
        "VariableNumFeaturesEncoder": var_num_feat,
        "LinearEncoder": linear,
        "FrequencyFeatureEncoder(optional)": freq,
    }

    for name, mod in modules.items():
        mod.post_init()
        _print_state_dict(name, mod)


if __name__ == "__main__":
    main()
