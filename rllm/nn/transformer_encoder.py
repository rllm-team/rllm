#  Copyright (c) Prior Labs GmbH 2025.

# NOTE: This layer intentionally exposes many knobs because it is reused in
# several TabPFN-style inference/training modes.
from __future__ import annotations

from typing import Any, ClassVar

import torch
from torch import nn
from torch.nn.modules.transformer import Module, Tensor

from rllm.nn.mlp import MLP
from .multi_head_attention import MultiHeadAttention


class PerFeatureEncoderLayer(Module):
    """Transformer encoder layer that processes each feature block separately.

    This layer consists of multi-head attention between features, multi-head
    attention between items, and feedforward neural networks (MLPs).

    It supports several configuration options.

    Args:
        d_model: The dimensionality of the input and output embeddings.
        nhead: The number of attention heads.
        dim_feedforward:
            The dimensionality of the feedforward network.
            Default is None (2 * d_model).
        activation: The activation function to use in the MLPs.
        layer_norm_eps: The epsilon value for layer normalization.
        pre_norm:
            If True, apply layer normalization before each sublayer
            (pre-norm); otherwise use post-norm.
        device: The device to use for the layer parameters.
        dtype: The data type to use for the layer parameters.
        recompute_attn: Whether to recompute attention during backpropagation.
        second_mlp: Whether to include a second MLP in the layer.
        layer_norm_with_elementwise_affine:
            Whether to use elementwise affine parameters in layer normalization.
        zero_init: Whether to initialize the output of the MLPs to zero.
        attention_between_features: Whether to apply attention between feature blocks.
        multiquery_item_attention: Whether to use multiquery attention for items.
        multiquery_item_attention_for_test_set:
            Whether to use multiquery attention for the test set.
        attention_init_gain: Gain used to initialize attention projections.
        two_sets_of_queries:
            Requires `multiquery_item_attention_for_test_set=True`.
        precomputed_kv:
            Reserved for compatibility with external checkpoints/APIs.
    """

    __constants__: ClassVar = ["batch_first"]

    def __init__(  # noqa: PLR0913
        self,
        *,
        d_model: int,
        nhead: int,
        dim_feedforward: int | None = None,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        pre_norm: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        recompute_attn: bool = False,
        second_mlp: bool = False,
        layer_norm_with_elementwise_affine: bool = False,
        zero_init: bool = False,
        attention_between_features: bool = True,
        multiquery_item_attention: bool = False,
        multiquery_item_attention_for_test_set: bool = False,
        two_sets_of_queries: bool = False,
        attention_init_gain: float = 1.0,
        precomputed_kv: None | torch.Tensor | tuple[torch.Tensor, torch.Tensor] = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        if multiquery_item_attention_for_test_set and multiquery_item_attention:
            raise ValueError(
                "Cannot use both multiquery_item_attention_for_test_set"
                "and multiquery_item_attention",
            )
        if two_sets_of_queries and not multiquery_item_attention_for_test_set:
            raise ValueError(
                "two_sets_of_queries requires multiquery_item_attention_for_test_set",
            )

        d_k = d_model // nhead
        d_v = d_model // nhead

        self.self_attn_between_features: MultiHeadAttention | None = None
        if attention_between_features:
            self.self_attn_between_features = MultiHeadAttention(
                input_size=d_model,
                output_size=d_model,
                d_k=d_k,
                d_v=d_v,
                nhead=nhead,
                device=device,
                dtype=dtype,
                initialize_output_to_zero=zero_init,
                recompute=recompute_attn,
                init_gain=attention_init_gain,
            )

        self.self_attn_between_items = MultiHeadAttention(
            input_size=d_model,
            output_size=d_model,
            d_k=d_k,
            d_v=d_v,
            nhead=nhead,
            device=device,
            dtype=dtype,
            share_kv_across_n_heads=nhead if multiquery_item_attention else 1,
            initialize_output_to_zero=zero_init,
            recompute=recompute_attn,
            init_gain=attention_init_gain,
        )

        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        self.mlp = MLP(
            dim=d_model,
            hidden_dim=dim_feedforward,
            activation=activation,
            device=device,
            dtype=dtype,
            initialize_output_to_zero=zero_init,
            recompute=recompute_attn,
        )
        self.layer_norms = nn.ModuleList(
            [
                nn.LayerNorm(
                    d_model,  # type: ignore
                    layer_norm_eps,
                    elementwise_affine=layer_norm_with_elementwise_affine,
                    **factory_kwargs,
                )
                for _ in range(4 if second_mlp else 3)
            ],
        )

        self.second_mlp: MLP | None = None
        if second_mlp:
            self.second_mlp = MLP(
                dim=d_model,
                hidden_dim=dim_feedforward,
                activation=activation,
                device=device,
                dtype=dtype,
                initialize_output_to_zero=zero_init,
                recompute=recompute_attn,
            )

        self.pre_norm = pre_norm
        self.recompute_attn = recompute_attn
        self.multiquery_item_attention_for_test_set = (
            multiquery_item_attention_for_test_set
        )
        self.two_sets_of_queries = two_sets_of_queries

    def forward(  # noqa: C901
        self,
        x: Tensor,
        single_eval_pos: int | None = None,
        *,
        cache_trainset_representation: bool = False,
        att_src: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """Run one encoder layer pass.

        Args:
            x:
                Input state of shape
                (batch_size, num_items, num_feature_blocks, d_model).
            single_eval_pos:
                Split position between train and test items. Items at indices
                `< single_eval_pos` are treated as train items.
            cache_trainset_representation:
                Whether to cache the train-set representation. When enabled,
                cached KV tensors may be reused for inference.
            att_src:
                Optional tensor used as the attention source for item attention,
                with shape
                (batch_size, num_train_items, num_feature_blocks, d_model).
                This is currently incompatible with
                `multiquery_item_attention_for_test_set` together with
                `cache_trainset_representation`.

        Returns:
            Updated transformer state with the same shape as `x`.
        """
        if single_eval_pos is None:
            single_eval_pos = 0

        layer_norm_idx = 0

        # MultiHeadAttention expects sequence dimension at `-2`.
        # For tabular data, this attention is applied between feature blocks.
        if self.self_attn_between_features is not None:
            if self.pre_norm:
                x = self.layer_norms[layer_norm_idx](x)
            x = self.self_attn_between_features(
                x,
                add_input=True,
                allow_inplace=True,
            )
            if not self.pre_norm:
                x = self.layer_norms[layer_norm_idx](x)
            layer_norm_idx += 1

        if self.second_mlp is not None:
            if self.pre_norm:
                x = self.layer_norms[layer_norm_idx](x)
            x = self.second_mlp(
                x,
                add_input=True,
                allow_inplace=True,
            )
            if not self.pre_norm:
                x = self.layer_norms[layer_norm_idx](x)
            layer_norm_idx += 1

        if self.pre_norm:
            x = self.layer_norms[layer_norm_idx](x)

        # MultiHeadAttention expects sequence dimension at `-2`.
        # For tabular data, this attention is applied between items.
        # Also, this is defaulted for all use cases.
        # When `multiquery_item_attention_for_test_set` is enabled,
        # the attention between items is always computed in multiquery mode,
        # but the test set and train set can be attended to separately.
        # When `att_src` is provided, it is used as the attention source for the entire batch instead of `x`.
        # This is currently incompatible with `multiquery_item_attention_for_test_set` together with `cache_trainset_representation`.
        if self.multiquery_item_attention_for_test_set:
            if single_eval_pos < x.shape[1]:
                new_x_test = self.self_attn_between_items(
                    x[:, single_eval_pos:].transpose(1, 2),
                    x[:, :single_eval_pos].transpose(1, 2) if single_eval_pos else None,
                    add_input=True,
                    allow_inplace=True,
                    reuse_first_head_kv=True,
                ).transpose(1, 2)
            else:
                new_x_test = None

            if single_eval_pos:
                new_x_train = self.self_attn_between_items(
                    x[:, :single_eval_pos].transpose(1, 2),
                    x[:, :single_eval_pos].transpose(1, 2),
                    add_input=True,
                    allow_inplace=True,
                ).transpose(1, 2)
            else:
                new_x_train = None

            x = torch.cat(
                [x_ for x_ in [new_x_train, new_x_test] if x_ is not None],
                dim=1,
            )
        else:
            attention_src_x = None
            if att_src is not None:
                attention_src_x = att_src.transpose(1, 2)
            elif single_eval_pos:
                attention_src_x = x[:, :single_eval_pos].transpose(1, 2)

            x = self.self_attn_between_items(
                x.transpose(1, 2),
                attention_src_x,
                add_input=True,
                allow_inplace=True,
            ).transpose(1, 2)

        if not self.pre_norm:
            x = self.layer_norms[layer_norm_idx](x)
        layer_norm_idx += 1

        if self.pre_norm:
            x = self.layer_norms[layer_norm_idx](x)
        x = self.mlp(
            x,
            add_input=True,
            allow_inplace=True,
        )
        if not self.pre_norm:
            x = self.layer_norms[layer_norm_idx](x)

        return x
