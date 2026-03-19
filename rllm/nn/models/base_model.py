import torch


class LinearClassifier(torch.nn.Module):
    """LayerNorm + Linear head for classification; outputs 1 logit if num_class<=2."""
    def __init__(self, num_class: int, hidden_dim: int = 128) -> None:
        super().__init__()
        out_dim = 1 if num_class <= 2 else num_class
        self.norm = torch.nn.LayerNorm(hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, hidden_dim) -> logits: (batch,) or (batch, num_class)."""
        return self.fc(self.norm(x))
