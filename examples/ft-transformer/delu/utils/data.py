"""An extension to `torch.utils.data`."""

from typing import Any, Tuple, TypeVar

from torch.utils.data import Dataset

from .._utils import deprecated

T = TypeVar('T')


__all__ = ['Enumerate', 'IndexDataset']


@deprecated('')
class Enumerate(Dataset):
    """Make a PyTorch dataset return indices in addition to items (like `enumerate`, but for datasets).

    <DEPRECATION MESSAGE>

    TL;DR:

    - ``dataset[i] -> value``
    - ``enumerated_dataset[i] -> (i, value)``

    **Usage**

    Creating the initial non-enumerated ``dataset``:

    >>> from torch.utils.data import DataLoader, TensorDataset
    >>>
    >>> X = torch.arange(10).float().view(5, 2)
    >>> X
    tensor([[0., 1.],
            [2., 3.],
            [4., 5.],
            [6., 7.],
            [8., 9.]])
    >>> Y = -10 * torch.arange(5)
    >>> Y
    tensor([  0, -10, -20, -30, -40])
    >>>
    >>> dataset = TensorDataset(X, Y)
    >>> dataset[2]
    (tensor([4., 5.]), tensor(-20))

    The enumerated dataset returns indices in addition to items:

    >>> enumerated_dataset = delu.utils.data.Enumerate(dataset)
    >>> enumerated_dataset[2]
    (2, (tensor([4., 5.]), tensor(-20)))
    >>>
    >>> for x_batch, y_batch in DataLoader(
    ...     dataset, batch_size=2
    ... ):
    ...     ...
    ...
    >>> for batch_idx, (x_batch, y_batch) in DataLoader(
    ...     enumerated_dataset, batch_size=2
    ... ):
    ...     print(batch_idx)
    tensor([0, 1])
    tensor([2, 3])
    tensor([4])

    The original dataset and its size remain accessible:

    >>> enumerated_dataset.dataset is dataset
    True
    >>> len(enumerated_dataset) == len(dataset)
    True
    """  # noqa: E501

    def __init__(self, dataset: Dataset, /) -> None:
        """
        Args:
            dataset: the original dataset.
        """
        self._dataset = dataset

    @property
    def dataset(self) -> Dataset:
        """The original dataset."""
        return self._dataset

    def __len__(self) -> int:
        """Get the length of the original dataset."""
        return len(self._dataset)  # type: ignore

    def __getitem__(self, index) -> Tuple[Any, Any]:
        """Return index and the corresponding item from the original dataset.

        Args:
            index: the index.
        Returns:
            (index, item)
        """
        return index, self._dataset[index]


@deprecated('')
class IndexDataset(Dataset):
    """A trivial dataset that yields indices back to user (useful for DistributedDataParallel (DDP)).

    <DEPRECATION MESSAGE>

    This simple dataset is useful when *both* conditions are true:

    1. A dataloader that yields batches of *indices* instead of *objects* is needed
    2. The `Distributed Data Parallel
       <https://pytorch.org/tutorials/intermediate/ddp_tutorial.html>`_ setup is used.

    .. note::
        If only the first condition is true, consider using the combinatation of
        `torch.randperm` and `torch.Tensor.split` instead.

    **Usage**

    >>> # doctest: +SKIP
    >>> from torch.utils.data import DataLoader
    >>> from torch.utils.data.distributed import DistributedSampler
    >>>
    >>> train_size = 1000
    >>> batch_size = 64
    >>> dataset = delu.data.IndexDataset(train_size)
    >>> # The dataset is really *that* trivial:
    >>> for i in range(train_size):
    ...     assert dataset[i] == i
    >>> dataloader = DataLoader(
    ...     dataset,
    ...     batch_size,
    ...     sampler=DistributedSampler(dataset),
    ... )
    >>> for epoch in range(n_epochs):
    ...     for batch_indices in dataloader:
    ...         ...
    """  # noqa: E501

    def __init__(self, size: int) -> None:
        """
        Args:
            size: the dataset size.
        """
        if size < 1:
            raise ValueError('size must be positive')
        self.size = size

    def __len__(self) -> int:
        """Get the dataset size."""
        return self.size

    def __getitem__(self, index: int) -> int:
        """Get the same index back.

        The index must be an integer from ``range(len(self))``.
        """
        # Some datasets support non-integer indices.
        if not isinstance(index, int):
            raise ValueError('index must be an integer')
        if index < 0 or index >= self.size:
            raise IndexError(
                f"The index {index} is out of range (the dataset size is {self.size})"
            )
        return index
