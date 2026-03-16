import torch


class LinearClassifier(torch.nn.Module):
    """Apply LayerNorm and a linear projection for classification.

    This lightweight head maps hidden representations to classification logits.
    For binary tasks it returns a single logit, while for multi-class tasks it
    returns one logit per class.

    Args:
        num_class (int): Number of prediction classes.
        hidden_dim (int): Dimensionality of the input hidden features.

    Returns:
        This class does not return tensors in ``__init__``.
        The ``forward`` method returns logits with shape ``[batch_size]`` or
        ``[batch_size, num_class]``.

    Example:
        >>> import torch
        >>> head = LinearClassifier(num_class=3, hidden_dim=16)
        >>> x = torch.randn(4, 16)
        >>> head(x).shape
        torch.Size([4, 3])
    """

    def __init__(self, num_class: int, hidden_dim: int = 128) -> None:
        super().__init__()
        out_dim = 1 if num_class <= 2 else num_class
        self.norm = torch.nn.LayerNorm(hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project hidden features into classification logits.

        Args:
            x (torch.Tensor): Input tensor of shape ``[batch_size, hidden_dim]``.

        Returns:
            torch.Tensor: Logits of shape ``[batch_size]`` for binary tasks or
            ``[batch_size, num_class]`` for multi-class tasks.

        Example:
            >>> import torch
            >>> head = LinearClassifier(num_class=2, hidden_dim=8)
            >>> x = torch.randn(3, 8)
            >>> head(x).shape
            torch.Size([3, 1])
        """
        return self.fc(self.norm(x))
