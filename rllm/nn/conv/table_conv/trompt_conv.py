from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch import Tensor
import torch.nn.functional as F

from rllm.nn.encoder.col_encoder import EmbeddingEncoder, LinearEncoder
from rllm.nn.encoder import TablePreEncoder
from rllm.types import ColType


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

    Example:
        >>> import torch
        >>> conv = TromptConv(in_dim=10, out_dim=16, num_prompts=4)
        >>> x = torch.randn(8, 10, 16)
        >>> x_prompt = torch.randn(8, 4, 16)
        >>> out = conv(x, x_prompt)
        >>> out.shape
        torch.Size([8, 4, 16])
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_prompts: int,
        metadata: Dict[ColType, List[Dict[str, Any]]] | None = None,
        num_groups: int = 2,
    ):
        super().__init__()
        self.num_prompts = num_prompts

        # Initialize pre-encoder with column-specific encoders
        col_encoder_dict = {
            ColType.CATEGORICAL: EmbeddingEncoder(
                post_module=torch.nn.LayerNorm(out_dim)
            ),
            ColType.NUMERICAL: LinearEncoder(
                in_dim=in_dim,
                post_module=torch.nn.Sequential(
                    torch.nn.ReLU(),
                    torch.nn.LayerNorm(out_dim),
                ),
            ),
        }
        self.feature_encoder = TablePreEncoder(
            out_dim=out_dim,
            metadata=metadata,
            col_encoder_dict=col_encoder_dict,
        )

        # Learnable parameters for feature importance and prompt embeddings
        self.emb_column = torch.nn.Parameter(torch.empty(in_dim, out_dim))
        self.emb_prompt = torch.nn.Parameter(torch.empty(num_prompts, out_dim))

        self.lin_se_prompt = torch.nn.Linear(out_dim * 2, out_dim)
        self.ln_column = torch.nn.LayerNorm(out_dim)
        self.ln_prompt = torch.nn.LayerNorm(out_dim)

        self.expand_weight = torch.nn.Parameter(torch.empty(num_prompts))
        self.group_norm = torch.nn.GroupNorm(
            num_groups=num_groups,
            num_channels=num_prompts,
        )

        self.reset_parameters()

    def reset_parameters(self):
        if self.feature_encoder is not None:
            self.feature_encoder.reset_parameters()

        torch.nn.init.xavier_uniform_(self.emb_column)
        torch.nn.init.xavier_uniform_(self.emb_prompt)
        torch.nn.init.xavier_uniform_(self.lin_se_prompt.weight)
        torch.nn.init.zeros_(self.lin_se_prompt.bias)
        torch.nn.init.uniform_(self.expand_weight)

        self.ln_column.reset_parameters()
        self.ln_prompt.reset_parameters()
        self.group_norm.reset_parameters()

    def forward(
        self,
        x: Tensor | Dict[ColType, Tensor],
        x_prompt: Tensor,
    ) -> Tensor:
        """Expand and aggregate feature embeddings conditioned on prompts.

        Args:
            x (Tensor | Dict[ColType, Tensor]): Input feature embeddings of shape
                ``[batch_size, in_dim, out_dim]`` or raw table feature dict.
            x_prompt (Tensor): Prompt embeddings of shape
                ``[batch_size, num_prompts, out_dim]``.

        Returns:
            Tensor: Aggregated prompt representations of shape
            ``[batch_size, num_prompts, out_dim]``.
        """
        if isinstance(x, dict):
            if self.feature_encoder is None:
                raise ValueError(
                    "Received raw feature dict but feature_encoder is not initialized. "
                    "Pass metadata when constructing TromptConv."
                )
            x = self.feature_encoder(x)

        emb_column = self.ln_column(self.emb_column)
        emb_prompt = self.ln_prompt(self.emb_prompt)

        # [num_prompts, out_dim] -> [batch_size, num_prompts, out_dim]
        se_prompt = emb_prompt.unsqueeze(0).repeat(x.size(0), 1, 1)
        # [batch_size, num_prompts, out_dim*2]
        se_prompt_cat = torch.cat([se_prompt, x_prompt], dim=-1)
        se_prompt_cat_hat = self.lin_se_prompt(se_prompt_cat) + se_prompt + x_prompt

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
