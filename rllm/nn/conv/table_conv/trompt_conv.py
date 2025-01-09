from __future__ import annotations
from typing import Union, Dict, List, Any

import torch
from torch import Tensor
import torch.nn.functional as F

from rllm.types import ColType
from rllm.nn.pre_encoder import FTTransformerPreEncoder


class TromptConv(torch.nn.Module):
    r"""The TromptConv Layer introduced in the
    `"Trompt: Towards a Better Deep Neural Network for Tabular Data"
    <https://arxiv.org/abs/2305.18446>`_ paper. Also it is konwn as TromptCell
    in the original paper.

    This layer first derives feature importance based on the
    `emb_column` and prompt embeddings `x_prompt`. Subsequently, it embeds
    the input features using a pre-encoder to obtain feature embeddings.
    Finally, it expands the features using the derived feature importance
    and the feature embeddings.

    Args:
        in_dim (int): Input dimensionality.
        out_dim (int): Output dimensionality, and hidden layer dimensionality.
        num_prompts (int): Number of prompts.
        num_groups (int): Number of groups for group normalization (default: 2).
        use_pre_encoder (bool): Whether to use a pre-encoder (default: :obj:`False`).
        metadata (Dict[ColType, List[Dict[str, Any]]], optional):
            Metadata for each column type, specifying the statistics and
            properties of the columns. (default: :obj:`None`).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_prompts: int,
        num_groups: int = 2,
        use_pre_encoder: bool = False,
        metadata: Dict[ColType, List[Dict[str, Any]]] = None,
    ):
        super().__init__()
        self.num_prompts = num_prompts

        self.emb_column = torch.nn.Parameter(torch.empty(in_dim, out_dim))
        self.emb_prompt = torch.nn.Parameter(torch.empty(num_prompts, out_dim))

        self.linear = torch.nn.Linear(out_dim * 2, out_dim)
        self.ln_column = torch.nn.LayerNorm(out_dim)
        self.ln_prompt = torch.nn.LayerNorm(out_dim)

        self.expand_weight = torch.nn.Parameter(torch.empty(num_prompts))
        self.group_norm = torch.nn.GroupNorm(
            num_groups=num_groups,
            num_channels=num_prompts,
        )

        # Define PreEncoder
        self.pre_encoder = None
        if use_pre_encoder:
            self.pre_encoder = FTTransformerPreEncoder(
                out_dim=out_dim,
                metadata=metadata,
            )

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.emb_column)
        torch.nn.init.xavier_uniform_(self.emb_prompt)
        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)
        torch.nn.init.uniform_(self.expand_weight)
        if self.pre_encoder is not None:
            self.pre_encoder.reset_parameters()

    def forward(self, x: Union[Dict, Tensor], x_prompt: Tensor) -> Tensor:
        if self.pre_encoder is not None:
            x = self.pre_encoder(x)

        emb_column = self.ln_column(self.emb_column)
        emb_prompt = self.ln_prompt(self.emb_prompt)

        # [num_prompts, out_dim] -> [batch_size, num_prompts, out_dim]
        se_prompt = emb_prompt.unsqueeze(0).repeat(x.size(0), 1, 1)
        # [batch_size, num_prompts, out_dim*2]
        se_prompt_cat = torch.cat([se_prompt, x_prompt], dim=-1)
        se_prompt_cat_hat = self.linear(se_prompt_cat) + se_prompt + x_prompt

        # [in_dim, out_dim] -> [batch_size, in_dim, out_dim]
        se_column = emb_column.unsqueeze(0).repeat(x_prompt.size(0), 1, 1)
        m_importance = torch.einsum("ijl,ikl->ijk", se_prompt_cat_hat, se_column)
        m_importance = F.softmax(m_importance, dim=-1)

        # [batch_size, num_prompts, in_dim, 1]
        m_importance = m_importance.unsqueeze(dim=-1)

        # [batch_size, in_dim, out_dim]
        # -> [batch_size, num_prompts, in_dim, out_dim]
        x_expand_weight = torch.einsum("ijl,k->ikjl", x, self.expand_weight)
        x_expand_weight = F.relu(x_expand_weight)
        x_expand_residual = x.unsqueeze(1).repeat(1, self.num_prompts, 1, 1)

        # Residual connection
        x = self.group_norm(x_expand_weight) + x_expand_residual

        x = (x * m_importance).sum(dim=2)
        return x
