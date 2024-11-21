from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


class TromptConv(torch.nn.Module):
    r"""The TromptConv Layer introduced in the
    `"Trompt: Towards a Better Deep Neural Network for Tabular Data"
    <https://arxiv.org/abs/2305.18446>`_ paper.

    Args:
        in_dim (int): Input dimensionality.
        hidden_dim (int): Hidden layer dimensionality.
        num_prompts (int): Number of prompts.
        num_groups (int): Number of groups for group normalization (default: 2).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_prompts: int,
        num_groups: int = 2,
    ):
        super().__init__()
        self.num_prompts = num_prompts

        self.emb_column = torch.nn.Parameter(torch.empty(in_dim, hidden_dim))
        self.emb_prompt = torch.nn.Parameter(torch.empty(num_prompts, hidden_dim))

        self.linear = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.ln_column = torch.nn.LayerNorm(hidden_dim)
        self.ln_prompt = torch.nn.LayerNorm(hidden_dim)

        self.expand_weight = torch.nn.Parameter(torch.empty(num_prompts))
        self.group_norm = torch.nn.GroupNorm(
            num_groups=num_groups,
            num_channels=num_prompts,
        )

        self.reset_parameters()

    def forward(self, x: Tensor, x_prompt: Tensor) -> Tensor:
        emb_column = self.ln_column(self.emb_column)
        emb_prompt = self.ln_prompt(self.emb_prompt)

        # [num_prompts, hidden_dim] -> [batch_size, num_prompts, hidden_dim]
        se_prompt = emb_prompt.unsqueeze(0).repeat(x.size(0), 1, 1)
        # [batch_size, num_prompts, hidden_dim*2]
        se_prompt_cat = torch.cat([se_prompt, x_prompt], dim=-1)
        se_prompt_cat_hat = self.linear(se_prompt_cat) + se_prompt + x_prompt

        # [in_dim, hidden_dim] -> [batch_size, in_dim, hidden_dim]
        se_column = emb_column.unsqueeze(0).repeat(x_prompt.size(0), 1, 1)
        m_importance = torch.einsum("ijl,ikl->ijk", se_prompt_cat_hat, se_column)
        m_importance = F.softmax(m_importance, dim=-1)

        # [batch_size, num_prompts, in_dim, 1]
        m_importance = m_importance.unsqueeze(dim=-1)

        # [batch_size, in_dim, hidden_dim]
        # -> [batch_size, num_prompts, in_dim, hidden_dim]
        x_expand_weight = torch.einsum("ijl,k->ikjl", x, self.expand_weight)
        x_expand_weight = F.relu(x_expand_weight)
        x_expand_residual = x.unsqueeze(1).repeat(1, self.num_prompts, 1, 1)
        # Residual connection
        x = self.group_norm(x_expand_weight) + x_expand_residual

        x = (x * m_importance).sum(dim=2)
        return x

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.emb_column)
        torch.nn.init.xavier_uniform_(self.emb_prompt)
        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)
        torch.nn.init.uniform_(self.expand_weight)
