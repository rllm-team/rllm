from typing import Any, Iterable, Callable, List, Dict

import torch
from torch import nn, Tensor

from rllm.types import ColType, NAMode
from rllm.nn.encoder.coltype_encoder import (
    CategoricalTransform,
    NumericalTransform
)
from rllm.nn.conv.tab_transformer_conv import MLP, TabTransformerConv


class TabTransformer(nn.Module):
    r"""The Tab-Transformer model introduced in the
    `"TabTransformer: Tabular Data Modeling Using Contextual Embeddings"
    <https://arxiv.org/abs/2012.06678>`_ paper.

    The model executes multi-layer column-interaction modeling exclusively
    on the categorical features. For numerical features, the model simply
    applies layer normalization on input features. The model utilizes an
    MLP(Multilayer Perceptron) for decoding.

    Args:
        hidden_dim (int): Hidden dimensionality for catrgorical features.
        layers (int): Number of convolution layers.
        heads (int): Number of heads in the self-attention layer.
        output_dim (int): Output dimensionality.
        head_dim (int): Dimensionality of each attention head.
        mlp_hidden_mults (Iterable[int]): Multiplicative coefficient
            applied to the hidden dimension of MLP.
        mlp_act (Callable): Activation function of MLP.
        attn_dropout (float): Dropout for Self-Attention.
        ff_dropout (float): Dropout for FeedForward layer.
        use_shared_categ_embedding (bool): Specifies whether to use
            shared category embeddings.
        shared_category_dim_divisor (int): Specifies shared category
            embedding dimension (floor division).
        col_stats_dict (Dict[
            :class:`rllm.types.ColType`,
            List[Dict[str, Any]]
        ):
            A dictionary that maps column type into stats. The column
            with same :class:`rllm.types.ColType` will be put together.
    """
    def __init__(
        self,
        *,
        hidden_dim: int,
        layers: int = 6,
        heads: int = 8,
        output_dim: int = 1,
        head_dim: int = 16,
        mlp_hidden_mults: Iterable[int] = (2,),
        mlp_act: Callable = None,
        attn_dropout: float = 0.,
        ff_dropout: float = 0.,
        use_shared_categ_embedding: bool = True,
        # in paper, they reserve dimension / 8 for category shared embedding
        shared_category_dim_divisor: int = 8.,
        col_stats_dict: Dict[ColType, List[Dict[str, Any]]] = None,
    ):
        super().__init__()
        if layers <= 0:
            raise ValueError(
                f"num_layers must be a positive integer (got {layers})"
            )
        self.hidden_dim = hidden_dim
        self.col_stats_dict = col_stats_dict
        self.use_shared_categ_embedding = use_shared_categ_embedding
        self.shared_category_dim_divisor = shared_category_dim_divisor
        # 1. Create column transforms
        self._prepare_column_transform()

        # 2. Prepare table convs backbone
        self.transformer = nn.ModuleList([
            TabTransformerConv(
                hidden_dim, heads,
                head_dim,
                attn_dropout,
                ff_dropout
            )
            for _ in range(layers)
        ])

        # 3. Prepare decoder
        input_size = (hidden_dim * self.categorical_col_len) + \
            self.numerical_col_len if \
            ColType.NUMERICAL in self.col_stats_dict.keys() \
            else (hidden_dim * self.categorical_col_len)
        hidden_dimensions = [input_size * t for t in mlp_hidden_mults]
        all_dimensions = [input_size, *hidden_dimensions, output_dim]
        self.decoder = MLP(all_dimensions, act=mlp_act)

        # reset parameters
        self.reset_parameters()

    def forward(
        self,
        feat_dict: Dict[ColType, Tensor],
        return_attns: bool = False
    ):
        # 1. Get embeddings by column transforms
        xs, post_softmax_attns = [], []
        if ColType.CATEGORICAL in self.col_stats_dict.keys():
            x_category = feat_dict[ColType.CATEGORICAL]
            category_embedding = self.category_transform(x_category)

            if self.use_shared_categ_embedding:
                shared_category_embedding = \
                    self.shared_category_embedding.unsqueeze(0).expand(
                        category_embedding.shape[0], -1, -1
                    )
                category_embedding = torch.cat(
                    (category_embedding, shared_category_embedding), dim=-1
                )
        # 2. Get representations table convs
            for layer in self.transformer:
                category_embedding, post_softmax_attn = layer(
                    category_embedding,
                    return_attn=True
                )
                post_softmax_attns.append(post_softmax_attn)
            attns = torch.stack(post_softmax_attns)

            flatten_category = category_embedding.reshape(
                category_embedding.size(0), -1
            )
            xs.append(flatten_category)

        if ColType.NUMERICAL in self.col_stats_dict.keys():
            x_numeric = feat_dict[ColType.NUMERICAL]
            numerical_embedding = self.numeric_transform(x_numeric)
            flatten_numeric = numerical_embedding.reshape(
                numerical_embedding.size(0), -1
            )
            xs.append(flatten_numeric)
        x = torch.cat(xs, dim=-1)
        # 3. Decode the representations
        logits = self.decoder(x)

        if return_attns:
            return logits, attns
        return logits

    def reset_parameters(self) -> None:
        if ColType.CATEGORICAL in self.col_stats_dict.keys():
            for layer in self.transformer:
                layer.reset_parameters()
        self.decoder.reset_parameters()

    def _prepare_column_transform(self):
        if ColType.CATEGORICAL in self.col_stats_dict.keys():
            categorical_stats_list = self.col_stats_dict[ColType.CATEGORICAL]
            self.categorical_col_len = len(categorical_stats_list)

            # create category transform
            shared_embedding_dim = 0 if not self.use_shared_categ_embedding \
                else int(self.hidden_dim // self.shared_category_dim_divisor)
            self.category_transform = CategoricalTransform(
                out_channels=self.hidden_dim - shared_embedding_dim,
                stats_list=categorical_stats_list,
                col_type=ColType.CATEGORICAL,
                na_mode=NAMode.MOST_FREQUENT,
            )
            self.category_transform.post_init()

            # create shared category embedding
            if self.use_shared_categ_embedding:
                self.shared_category_embedding = nn.Parameter(
                    torch.zeros(self.categorical_col_len, shared_embedding_dim)
                )
                nn.init.normal_(self.shared_category_embedding, std=0.02)

        if ColType.NUMERICAL in self.col_stats_dict.keys():
            numerical_stats_list = self.col_stats_dict[ColType.NUMERICAL]
            self.numerical_col_len = len(numerical_stats_list)

            # create numerical transform
            self.numeric_transform = NumericalTransform(
                type='stack',
                out_channels=1,
                stats_list=numerical_stats_list,
                col_type=ColType.NUMERICAL,
                na_mode=NAMode.MEAN,
            )
            self.numeric_transform.post_init()
