import torch


class LinearClassifier(torch.nn.Module):
    r"""LinearClassifier: Simple linear classification head with
    layer normalization followed by a fully-connected layer on the
    CLS embedding.

    Args:
        num_class (int): Number of target classes (<=2 produces a single logit).
        hidden_dim (int): Dimensionality of input CLS embedding.
    """

    def __init__(self, num_class: int, hidden_dim: int = 128) -> None:
        super().__init__()
        out_dim = 1 if num_class <= 2 else num_class
        self.fc = torch.nn.Linear(hidden_dim, out_dim)
        self.norm = torch.nn.LayerNorm(hidden_dim)

    def forward(self, cls_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cls_emb (torch.Tensor): CLS token embeddings of shape (batch_size, hidden_dim).

        Returns:
            torch.Tensor: Classification logits of shape
                (batch_size,) for binary or (batch_size, num_class).
        """
        x = self.norm(cls_emb)
        logits = self.fc(x)
        return logits
