from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


ModelVersion = Literal["v2_6"]
ExportMode = Literal["icl", "mlp_stub", "tree_stub"]


@dataclass
class TabPFNVersionConfig:
    """Runtime config for the retained TabPFN inference wrapper."""

    model_version: ModelVersion = "v2_6"
    n_layers_cls: int | None = 24
    n_layers_reg: int | None = 24
    feature_group_size: int | None = 3
    use_regression_mlp_encoder: bool = False
    regression_mlp_hidden_dim: int = 1024
    use_thinking_rows: bool = True
    n_thinking_rows: int = 64
    preprocessing_recipe: str = "v2_6_default"
    use_robust_scaling: bool = False
    use_soft_clipping: bool = False
    use_quantile_transform: bool = False
    use_standard_scaling: bool = False
    enable_temperature_scaling: bool = False
    enable_threshold_tuning: bool = False
    export_mode: ExportMode = "icl"
    enable_flash_attention: bool = False
    checkpoint_version: str = "tabpfn_rllm_v2_6_compat"

    def resolve_nlayers(self, model_type: Literal["clf", "reg"], fallback: int) -> int:
        if model_type == "clf":
            return int(self.n_layers_cls if self.n_layers_cls is not None else fallback)
        return int(self.n_layers_reg if self.n_layers_reg is not None else fallback)

    def resolve_feature_group_size(self, fallback: int) -> int:
        value = self.feature_group_size if self.feature_group_size is not None else fallback
        return max(1, int(value))

    @classmethod
    def from_version(cls, version: ModelVersion) -> "TabPFNVersionConfig":
        if version != "v2_6":
            raise ValueError(
                f"Unsupported TabPFN version: {version}. Only 'v2_6' is available."
            )
        return cls(
            model_version="v2_6",
            n_layers_cls=24,
            n_layers_reg=24,
            feature_group_size=3,
            use_regression_mlp_encoder=True,
            regression_mlp_hidden_dim=1024,
            use_thinking_rows=True,
            n_thinking_rows=64,
            preprocessing_recipe="v2_6_default",
            checkpoint_version="tabpfn_rllm_v2_6_compat",
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_version_config(
    model_type: Literal["clf", "reg"],
    *,
    model_version: ModelVersion = "v2_6",
    n_layers_cls: int | None = None,
    n_layers_reg: int | None = None,
    feature_group_size: int | None = None,
    use_regression_mlp_encoder: bool | None = None,
    regression_mlp_hidden_dim: int | None = None,
    use_thinking_rows: bool | None = None,
    n_thinking_rows: int | None = None,
    preprocessing_recipe: str | None = None,
    use_robust_scaling: bool | None = None,
    use_soft_clipping: bool | None = None,
    use_quantile_transform: bool | None = None,
    use_standard_scaling: bool | None = None,
    enable_temperature_scaling: bool | None = None,
    enable_threshold_tuning: bool | None = None,
    export_mode: ExportMode = "icl",
    enable_flash_attention: bool = False,
) -> TabPFNVersionConfig:
    """Factory used by public constructors with retained runtime defaults."""
    preset = TabPFNVersionConfig.from_version(model_version)
    if preprocessing_recipe is None:
        preprocessing_recipe = preset.preprocessing_recipe
    if use_robust_scaling is None:
        use_robust_scaling = preset.use_robust_scaling
    if use_soft_clipping is None:
        use_soft_clipping = preset.use_soft_clipping
    if use_quantile_transform is None:
        use_quantile_transform = preset.use_quantile_transform
    if use_standard_scaling is None:
        use_standard_scaling = preset.use_standard_scaling
    if enable_temperature_scaling is None:
        enable_temperature_scaling = preset.enable_temperature_scaling
    if enable_threshold_tuning is None:
        enable_threshold_tuning = preset.enable_threshold_tuning
    if use_regression_mlp_encoder is None:
        use_regression_mlp_encoder = preset.use_regression_mlp_encoder
    if regression_mlp_hidden_dim is None:
        regression_mlp_hidden_dim = preset.regression_mlp_hidden_dim
    if use_thinking_rows is None:
        use_thinking_rows = preset.use_thinking_rows
    if n_thinking_rows is None:
        n_thinking_rows = preset.n_thinking_rows
    cfg = TabPFNVersionConfig(
        model_version=model_version,
        n_layers_cls=n_layers_cls if n_layers_cls is not None else preset.n_layers_cls,
        n_layers_reg=n_layers_reg if n_layers_reg is not None else preset.n_layers_reg,
        feature_group_size=(
            feature_group_size
            if feature_group_size is not None
            else preset.feature_group_size
        ),
        use_regression_mlp_encoder=use_regression_mlp_encoder,
        regression_mlp_hidden_dim=regression_mlp_hidden_dim,
        use_thinking_rows=use_thinking_rows,
        n_thinking_rows=n_thinking_rows,
        preprocessing_recipe=preprocessing_recipe,
        use_robust_scaling=use_robust_scaling,
        use_soft_clipping=use_soft_clipping,
        use_quantile_transform=use_quantile_transform,
        use_standard_scaling=use_standard_scaling,
        enable_temperature_scaling=enable_temperature_scaling,
        enable_threshold_tuning=enable_threshold_tuning,
        export_mode=export_mode,
        enable_flash_attention=enable_flash_attention,
    )
    if model_type == "clf":
        cfg.use_regression_mlp_encoder = False
    return cfg
