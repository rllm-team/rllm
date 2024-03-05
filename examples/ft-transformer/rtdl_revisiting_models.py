"""Revisiting Deep Learning Models for Tabular Data."""

__version__ = '0.0.2'

__all__ = [
    'LinearEmbeddings',
    'CategoricalEmbeddings',
    'MultiheadAttention',
    'FTTransformerBackbone',
    'FTTransformer',
]

import itertools
import math
import typing
import warnings
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch import Tensor
from torch.nn.parameter import Parameter

_INTERNAL_ERROR = 'Internal error'


def _named_sequential(*modules) -> nn.Sequential:
    return nn.Sequential(OrderedDict(modules))


class LinearEmbeddings(nn.Module):
    """Linear embeddings for continuous features.

    **Shape**

    - Input: `(*, n_features)`
    - Output: `(*, n_features, d_embedding)`

    **Examples**

    >>> batch_size = 2
    >>> n_cont_features = 3
    >>> x = torch.randn(batch_size, n_cont_features)
    >>> d_embedding = 4
    >>> m = LinearEmbeddings(n_cont_features, d_embedding)
    >>> m(x).shape
    torch.Size([2, 3, 4])
    """

    def __init__(self, n_features: int, d_embedding: int) -> None:
        """
        Args:
            n_features: the number of continous features.
            d_embedding: the embedding size.
        """
        if n_features <= 0:
            raise ValueError(f'n_features must be positive, however: {n_features=}')
        if d_embedding <= 0:
            raise ValueError(f'd_embedding must be positive, however: {d_embedding=}')

        super().__init__()
        self.weight = Parameter(torch.empty(n_features, d_embedding))
        self.bias = Parameter(torch.empty(n_features, d_embedding))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        d_rqsrt = self.weight.shape[1] ** -0.5
        nn.init.uniform_(self.weight, -d_rqsrt, d_rqsrt)
        nn.init.uniform_(self.bias, -d_rqsrt, d_rqsrt)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim < 2:
            raise ValueError(
                f'The input must have at least two dimensions, however: {x.ndim=}'
            )

        x = x[..., None] * self.weight
        x = x + self.bias[None]
        return x


class CategoricalEmbeddings(nn.Module):
    """Embeddings for categorical features.

    **Examples**

    >>> cardinalities = [3, 10]
    >>> x = torch.tensor([
    ...     [0, 5],
    ...     [1, 7],
    ...     [0, 2],
    ...     [2, 4]
    ... ])
    >>> x.shape  # (batch_size, n_cat_features)
    torch.Size([4, 2])
    >>> m = CategoricalEmbeddings(cardinalities, d_embedding=5)
    >>> m(x).shape  # (batch_size, n_cat_features, d_embedding)
    torch.Size([4, 2, 5])
    """

    def __init__(
        self, cardinalities: List[int], d_embedding: int, bias: bool = True
    ) -> None:
        """
        Args:
            cardinalities: the number of distinct values for each feature.
            d_embedding: the embedding size.
            bias: if `True`, for each feature, a trainable vector is added to the
                embedding regardless of a feature value. For each feature, a separate
                non-shared bias vector is allocated.
                In the paper, FT-Transformer uses `bias=True`.
        """
        super().__init__()
        if not cardinalities:
            raise ValueError('cardinalities must not be empty')
        if any(x <= 0 for x in cardinalities):
            i, value = next((i, x) for i, x in enumerate(cardinalities) if x <= 0)
            raise ValueError(
                'cardinalities must contain only positive values,'
                f' however: cardinalities[{i}]={value}'
            )
        if d_embedding <= 0:
            raise ValueError(f'd_embedding must be positive, however: {d_embedding=}')

        self.embeddings = nn.ModuleList(
            [nn.Embedding(x, d_embedding) for x in cardinalities]
        )
        self.bias = (
            Parameter(torch.empty(len(cardinalities), d_embedding)) if bias else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        d_rsqrt = self.embeddings[0].embedding_dim ** -0.5
        for m in self.embeddings:
            nn.init.uniform_(m.weight, -d_rsqrt, d_rsqrt)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -d_rsqrt, d_rsqrt)

    def forward(self, x: Tensor) -> Tensor:
        """Do the forward pass."""
        if x.ndim < 2:
            raise ValueError(
                f'The input must have at least two dimensions, however: {x.ndim=}'
            )
        n_features = len(self.embeddings)
        if x.shape[-1] != n_features:
            raise ValueError(
                'The last input dimension (the number of categorical features) must be'
                ' equal to the number of cardinalities passed to the constructor.'
                f' However: {x.shape[-1]=}, len(cardinalities)={n_features}'
            )

        x = torch.stack(
            [self.embeddings[i](x[..., i]) for i in range(n_features)], dim=-2
        )
        if self.bias is not None:
            x = x + self.bias
        return x


_LINFORMER_KV_COMPRESSION_SHARING = Literal['headwise', 'key-value']


class MultiheadAttention(nn.Module):
    """Multihead (Self-/Cross-)Attention with an optional linear attention from ["Linformer: Self-Attention with Linear Complexity"](https://arxiv.org/abs/2006.04768).

    **Examples**

    >>> batch_size, n_tokens, d_embedding = 2, 3, 16
    >>> n_heads = 8
    >>> a = torch.randn(batch_size, n_tokens, d_embedding)
    >>> b = torch.randn(batch_size, n_tokens * 2, d_embedding)
    >>> m = MultiheadAttention(
    ...     d_embedding=d_embedding, n_heads=n_heads, dropout=0.2
    >>> )
    >>>
    >>> # Self-attention.
    >>> assert m(a, a).shape == a.shape
    >>>
    >>> # Cross-attention.
    >>> assert m(a, b).shape == a.shape
    >>>
    >>> # Linformer attention.
    >>> m = MultiheadAttention(
    ...     d_embedding=d_embedding,
    ...     n_heads=n_heads,
    ...     dropout=0.2,
    ...     n_tokens=n_tokens,
    ...     linformer_kv_compression_ratio=0.5,
    ...     linformer_kv_compression_sharing='headwise',
    >>> )
    >>> assert m(a, a).shape == a.shape
    """  # noqa: E501

    def __init__(
        self,
        *,
        d_embedding: int,
        n_heads: int,
        dropout: float,
        # Linformer arguments.
        n_tokens: Optional[int] = None,
        linformer_kv_compression_ratio: Optional[float] = None,
        linformer_kv_compression_sharing: Optional[
            _LINFORMER_KV_COMPRESSION_SHARING
        ] = None,
    ) -> None:
        """
        Args:
            d_embedding: the embedding size for one token.
                Must be a multiple of `n_heads`.
            n_heads: the number of heads. If greater than 1, then the module will have
                an additional output layer (the so called "mixing" layer).
            dropout: the dropout rate for the attention probability map.
            n_tokens: the number of tokens
                (must be provided if `linformer_kv_compression_ratio` is not None)
            linformer_kv_compression_ratio: Linformer-style compression rate.
                Must be within the interval `(0.0, 1.0)`.
            linformer_kv_compression_sharing: Linformer compression sharing policy.
                Must be provided if `linformer_kv_compression_ratio` is not None.
                (non-shared Linformer compression is not supported; the "layerwise"
                sharing policy is not supported).
        """
        if n_heads < 1:
            raise ValueError(f'n_heads must be positive, however: {n_heads=}')
        if d_embedding % n_heads:
            raise ValueError(
                'd_embedding must be a multiple of n_heads,'
                f' however: {d_embedding=}, {n_heads=}'
            )

        super().__init__()
        self.W_q = nn.Linear(d_embedding, d_embedding)
        self.W_k = nn.Linear(d_embedding, d_embedding)
        self.W_v = nn.Linear(d_embedding, d_embedding)
        self.W_out = nn.Linear(d_embedding, d_embedding) if n_heads > 1 else None
        self.dropout = nn.Dropout(dropout) if dropout else None
        self._n_heads = n_heads

        if linformer_kv_compression_ratio is not None:
            if n_tokens is None:
                raise ValueError(
                    'If linformer_kv_compression_ratio is not None,'
                    ' then n_tokens also must not be None'
                )
            if linformer_kv_compression_sharing not in typing.get_args(
                _LINFORMER_KV_COMPRESSION_SHARING
            ):
                raise ValueError(
                    'Valid values of linformer_kv_compression_sharing include:'
                    f' {typing.get_args(_LINFORMER_KV_COMPRESSION_SHARING)},'
                    f' however: {linformer_kv_compression_sharing=}'
                )
            if (
                linformer_kv_compression_ratio <= 0.0
                or linformer_kv_compression_ratio >= 1.0
            ):
                raise ValueError(
                    'linformer_kv_compression_ratio must be from the open interval'
                    f' (0.0, 1.0), however: {linformer_kv_compression_ratio=}'
                )

            def make_linformer_kv_compression():
                return nn.Linear(
                    n_tokens,
                    max(int(n_tokens * linformer_kv_compression_ratio), 1),
                    bias=False,
                )

            self.key_compression = make_linformer_kv_compression()
            self.value_compression = (
                make_linformer_kv_compression()
                if linformer_kv_compression_sharing == 'headwise'
                else None
            )
        else:
            if n_tokens is not None:
                raise ValueError(
                    'If linformer_kv_compression_ratio is None,'
                    ' then n_tokens also must be None'
                )
            if linformer_kv_compression_sharing is not None:
                raise ValueError(
                    'If linformer_kv_compression_ratio is None,'
                    ' then linformer_kv_compression_sharing also must be None'
                )
            self.key_compression = None
            self.value_compression = None

        for m in [self.W_q, self.W_k, self.W_v]:
            nn.init.zeros_(m.bias)
        if self.W_out is not None:
            nn.init.zeros_(self.W_out.bias)

    def _reshape(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, d = x.shape
        d_head = d // self._n_heads
        return (
            x.reshape(batch_size, n_tokens, self._n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self._n_heads, n_tokens, d_head)
        )

    def forward(self, x_q: Tensor, x_kv: Tensor) -> Tensor:
        """Do the forward pass."""
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        if self.key_compression is not None:
            k = self.key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = (
                self.key_compression
                if self.value_compression is None
                else self.value_compression
            )(v.transpose(1, 2)).transpose(1, 2)

        batch_size = len(q)
        d_head_key = k.shape[-1] // self._n_heads
        d_head_value = v.shape[-1] // self._n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)
        attention_logits = q @ k.transpose(1, 2) / math.sqrt(d_head_key)
        attention_probs = F.softmax(attention_logits, dim=-1)
        if self.dropout is not None:
            attention_probs = self.dropout(attention_probs)
        x = attention_probs @ self._reshape(v)
        x = (
            x.reshape(batch_size, self._n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self._n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)
        return x


class _ReGLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] % 2:
            raise ValueError(
                'For the ReGLU activation, the last input dimension'
                f' must be a multiple of 2, however: {x.shape[-1]=}'
            )
        a, b = x.chunk(2, dim=-1)
        return a * F.relu(b)


_TransformerFFNActivation = Literal['ReLU', 'ReGLU']


class FTTransformerBackbone(nn.Module):
    """The backbone of FT-Transformer.

    The differences with Transformer from the paper
    ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) are as follows:

    - the so called "PreNorm" variation is used
        (`norm_first=True` in terms of `torch.nn.TransformerEncoderLayer`)
    - the very first normalization is skipped. This is **CRUCIAL** for FT-Transformer
        in the PreNorm configuration.

    **Examples**

    >>> batch_size = 2
    >>> n_tokens = 3
    >>> d_block = 16
    >>> x = torch.randn(batch_size, n_tokens, d_block)
    >>> d_out = 1
    >>> m = FTTransformerBackbone(
    ...     d_out=d_out,
    ...     n_blocks=2,
    ...     d_block=d_block,
    ...     attention_n_heads=8,
    ...     attention_dropout=0.2,
    ...     ffn_d_hidden=None,
    ...     ffn_d_hidden_multiplier=2.0,
    ...     ffn_dropout=0.1,
    ...     residual_dropout=0.0,
    ... )
    >>> m(x).shape
    torch.Size([2, 1])
    """

    def __init__(
        self,
        *,
        d_out: Optional[int],
        n_blocks: int,
        d_block: int,
        attention_n_heads: int,
        attention_dropout: float,
        ffn_d_hidden: Optional[int] = None,
        ffn_d_hidden_multiplier: Optional[float],
        ffn_dropout: float,
        # NOTE[DIFF]
        # In the paper, FT-Transformer uses the ReGLU activation.
        # Here, to illustrate the difference, ReLU activation is also supported
        # (in particular, see the docstring).
        ffn_activation: _TransformerFFNActivation = 'ReGLU',
        residual_dropout: float,
        n_tokens: Optional[int] = None,
        linformer_kv_compression_ratio: Optional[float] = None,
        linformer_kv_compression_sharing: Optional[
            _LINFORMER_KV_COMPRESSION_SHARING
        ] = None,
    ):
        """
        Args:
            d_out: the output size.
            n_blocks: the number of blocks.
            d_block: the block width
                (or, equivalently, the embedding size of each feature).
                Must be a multiple of `attention_n_heads`.
            attention_n_heads: the number of attention heads in `MultiheadAttention`.
            attention_dropout: the dropout rate in `MultiheadAttention`. Usually,
                positive values work better, even if the number of features is low.
            ffn_d_hidden: the hidden representation size after the activation in the
                feed-forward blocks (or, equivalently, the *input* size of the *second*
                linear layer in the feed-forward blocks). If ``ffn_use_reglu``
                is `True`, then the *output* size of the *first* linear layer
                will be set to ``2 * ffn_d_hidden``.
            ffn_d_hidden_multiplier: the alternative way to set `ffn_d_hidden` as
                `int(d_block * ffn_d_hidden_multiplier)`.
            ffn_dropout: the dropout rate for the hidden representation
                in the feed-forward blocks.
            ffn_activation: the activation used in the FFN blocks. To maintain (almost)
                the same number of parameters between different activations:
                <ffn_d_hidden_multiplier for ReGLU> = <2 / 3 * ffn_d_hidden_multiplier for ReLU>
                or
                <ffn_d_hidden_multiplier for ReLU> = <3 / 2 * ffn_d_hidden_multiplier for ReGLU>
            residual_dropout: the dropout rate for all residual branches.
            n_tokens: the argument for `MultiheadAttention`.
            linformer_kv_compression_ratio: the argument for `MultiheadAttention`.
            linformer_kv_compression_sharing: the argument for `MultiheadAttention`.
        """  # noqa: E501
        if ffn_activation not in typing.get_args(_TransformerFFNActivation):
            raise ValueError(
                'ffn_activation must be one of'
                f' {typing.get_args(_TransformerFFNActivation)}.'
                f' However: {ffn_activation=}'
            )
        if ffn_d_hidden is None:
            if ffn_d_hidden_multiplier is None:
                raise ValueError(
                    'If ffn_d_hidden is None,'
                    ' then ffn_d_hidden_multiplier must not be None'
                )
            ffn_d_hidden = int(d_block * cast(float, ffn_d_hidden_multiplier))
        else:
            if ffn_d_hidden_multiplier is not None:
                raise ValueError(
                    'If ffn_d_hidden is not None,'
                    ' then ffn_d_hidden_multiplier must be None'
                )

        super().__init__()
        ffn_use_reglu = ffn_activation == 'ReGLU'
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        # >>> attention
                        'attention': MultiheadAttention(
                            d_embedding=d_block,
                            n_heads=attention_n_heads,
                            dropout=attention_dropout,
                            n_tokens=n_tokens,
                            linformer_kv_compression_ratio=linformer_kv_compression_ratio,
                            linformer_kv_compression_sharing=linformer_kv_compression_sharing,
                        ),
                        'attention_residual_dropout': nn.Dropout(residual_dropout),
                        # >>> feed-forward
                        'ffn_normalization': nn.LayerNorm(d_block),
                        'ffn': _named_sequential(
                            (
                                'linear1',
                                # ReGLU divides dimension by 2,
                                # so multiplying by 2 to compensate for this.
                                nn.Linear(
                                    d_block, ffn_d_hidden * (2 if ffn_use_reglu else 1)
                                ),
                            ),
                            ('activation', _ReGLU() if ffn_use_reglu else nn.ReLU()),
                            ('dropout', nn.Dropout(ffn_dropout)),
                            ('linear2', nn.Linear(ffn_d_hidden, d_block)),
                        ),
                        'ffn_residual_dropout': nn.Dropout(residual_dropout),
                        # >>> output (for hook-based introspection)
                        'output': nn.Identity(),
                        # >>> the very first normalization
                        **(
                            {}
                            if layer_idx == 0
                            else {'attention_normalization': nn.LayerNorm(d_block)}
                        ),
                    }
                )
                for layer_idx in range(n_blocks)
            ]
        )
        self.output = (
            None
            if d_out is None
            else _named_sequential(
                ('normalization', nn.LayerNorm(d_block)),
                ('activation', nn.ReLU()),
                ('linear', nn.Linear(d_block, d_out)),
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        """Do the forward pass."""
        if x.ndim != 3:
            raise ValueError(
                f'The input must have exactly three dimension, however: {x.ndim=}'
            )

        n_blocks = len(self.blocks)
        for i_block, block in enumerate(self.blocks):
            block = cast(nn.ModuleDict, block)

            x_identity = x
            if 'attention_normalization' in block:
                x = block['attention_normalization'](x)
            x = block['attention'](x[:, :1] if i_block + 1 == n_blocks else x, x)
            x = block['attention_residual_dropout'](x)
            x = x_identity + x

            x_identity = x
            x = block['ffn_normalization'](x)
            x = block['ffn'](x)
            x = block['ffn_residual_dropout'](x)
            x = x_identity + x

            x = block['output'](x)

        x = x[:, 0]  # The representation of [CLS]-token.

        if self.output is not None:
            x = self.output(x)
        return x


class _CLSEmbedding(nn.Module):
    def __init__(self, d_embedding: int) -> None:
        super().__init__()
        self.weight = Parameter(torch.empty(d_embedding))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        d_rsqrt = self.weight.shape[-1] ** -0.5
        nn.init.uniform_(self.weight, -d_rsqrt, d_rsqrt)

    def forward(self, batch_dims: Tuple[int]) -> Tensor:
        if not batch_dims:
            raise ValueError('The input must be non-empty')

        return self.weight.expand(*batch_dims, 1, -1)


class FTTransformer(nn.Module):
    """The FT-Transformer model from Section 3.3 in the paper."""

    def __init__(
        self,
        *,
        n_cont_features: int,
        cat_cardinalities: List[int],
        _is_default: bool = False,
        **backbone_kwargs,
    ) -> None:
        """
        Args:
            n_cont_features: the number of continuous features.
            cat_cardinalities: the cardinalities of categorical features.
                Pass en empty list if there are no categorical features.
            _is_default: this is a technical argument, don't set it manually.
            backbone_kwargs: the keyword arguments for the `FTTransformerBackbone`.
        """
        if n_cont_features < 0:
            raise ValueError(
                f'n_cont_features must be non-negative, however: {n_cont_features=}'
            )
        if n_cont_features == 0 and not cat_cardinalities:
            raise ValueError(
                'At least one type of features must be presented, however:'
                f' {n_cont_features=}, {cat_cardinalities=}'
            )
        if 'n_tokens' in backbone_kwargs:
            raise ValueError(
                'backbone_kwargs must not contain key "n_tokens"'
                ' (the number of tokens will be inferred automatically)'
            )

        super().__init__()
        d_block: int = backbone_kwargs['d_block']
        self.cls_embedding = _CLSEmbedding(d_block)

        # >>> Feature embeddings (Figure 2a in the paper).
        self.cont_embeddings = (
            LinearEmbeddings(n_cont_features, d_block) if n_cont_features > 0 else None
        )
        self.cat_embeddings = (
            CategoricalEmbeddings(cat_cardinalities, d_block, True)
            if cat_cardinalities
            else None
        )
        # <<<

        self.backbone = FTTransformerBackbone(
            **backbone_kwargs,
            n_tokens=(
                None
                if backbone_kwargs.get('linformer_kv_compression_ratio') is None
                else 1 + n_cont_features + len(cat_cardinalities)
            ),
        )
        self._is_default = _is_default

    @classmethod
    def get_default_kwargs(cls, n_blocks: int = 3) -> Dict[str, Any]:
        """Get the default hyperparameters.

        Args:
            n_blocks: the number of blocks. The supported values are: 1, 2, 3, 4, 5, 6.
        Returns:
            the default keyword arguments for the constructor.
        """
        if n_blocks < 0 or n_blocks > 6:
            raise ValueError(
                'Default configurations are available'
                ' only for the following values of n_blocks: 1, 2, 3, 4, 5, 6.'
                f' However, {n_blocks=}'
            )
        return {
            'n_blocks': n_blocks,
            'd_block': [96, 128, 192, 256, 320, 384][n_blocks - 1],
            'attention_n_heads': 8,
            'attention_dropout': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35][n_blocks - 1],
            'ffn_d_hidden': None,
            # "4 / 3" for ReGLU leads to almost the same number of parameters
            # as "2.0" for ReLU.
            'ffn_d_hidden_multiplier': 4 / 3,
            'ffn_dropout': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25][n_blocks - 1],
            'residual_dropout': 0.0,
            '_is_default': True,
        }

    def make_parameter_groups(self) -> List[Dict[str, Any]]:
        """Make parameter groups for optimizers.

        The difference with calling this method instead of
        `.parameters()` is that this method always sets `weight_decay=0.0`
        for some of the parameters.

        Returns:
            the parameter groups that can be passed to PyTorch optimizers.
        """

        def get_parameters(m: Optional[nn.Module]) -> Iterable[Parameter]:
            return () if m is None else m.parameters()

        zero_wd_group: Dict[str, Any] = {
            'params': set(
                itertools.chain(
                    get_parameters(self.cls_embedding),
                    get_parameters(self.cont_embeddings),
                    get_parameters(self.cat_embeddings),
                    itertools.chain.from_iterable(
                        m.parameters()
                        for block in self.backbone.blocks
                        for name, m in block.named_children()
                        if name.endswith('_normalization')
                    ),
                    (
                        p
                        for name, p in self.named_parameters()
                        if name.endswith('.bias')
                    ),
                )
            ),
            'weight_decay': 0.0,
        }
        main_group: Dict[str, Any] = {
            'params': [p for p in self.parameters() if p not in zero_wd_group['params']]
        }
        zero_wd_group['params'] = list(zero_wd_group['params'])
        return [main_group, zero_wd_group]

    def make_default_optimizer(self) -> torch.optim.AdamW:
        """Create the "default" `torch.nn.AdamW` suitable for the *default* FT-Transformer.

        Returns:
            the optimizer.
        """  # noqa: E501
        if not self._is_default:
            warnings.warn(
                'The default opimizer is supposed to be used in a combination'
                ' with the default FT-Transformer.'
            )
        return torch.optim.AdamW(
            self.make_parameter_groups(), lr=1e-4, weight_decay=1e-5
        )

    _FORWARD_BAD_ARGS_MESSAGE = (
        'Based on the arguments passed to the constructor of FTTransformer, {}'
    )

    def forward(self, x_cont: Optional[Tensor], x_cat: Optional[Tensor]) -> Tensor:
        """Do the forward pass."""
        x_any = x_cat if x_cont is None else x_cont
        if x_any is None:
            raise ValueError('At least one of x_cont and x_cat must be provided.')

        x_embeddings: List[Tensor] = []
        if self.cls_embedding is not None:
            x_embeddings.append(self.cls_embedding(x_any.shape[:-1]))

        for argname, argvalue, module in [
            ('x_cont', x_cont, self.cont_embeddings),
            ('x_cat', x_cat, self.cat_embeddings),
        ]:
            if module is None:
                if argvalue is not None:
                    raise ValueError(
                        FTTransformer._FORWARD_BAD_ARGS_MESSAGE.format(
                            f'{argname} must be None'
                        )
                    )
            else:
                if argvalue is None:
                    raise ValueError(
                        FTTransformer._FORWARD_BAD_ARGS_MESSAGE.format(
                            f'{argname} must not be None'
                        )
                    )
                x_embeddings.append(module(argvalue))
        assert x_embeddings, _INTERNAL_ERROR
        x = torch.cat(x_embeddings, dim=1)
        x = self.backbone(x)
        return x
