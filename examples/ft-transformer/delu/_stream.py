import math
from copy import deepcopy
from typing import Any, Dict, Iterable, Iterator, Optional, Sized, Union

from ._utils import deprecated

_DEFAULT_PROGRESS_BAR_CONFIG: Dict[str, Any] = {}


def _try_len(x):
    try:
        return len(x)
    except (TypeError, NotImplementedError):
        return None


@deprecated('')
class Stream:
    """Smart wrapper for data loaders and iterables.

    <DEPRECATION MESSAGE>

    `Stream` simplifies managing loops, especially in typical Deep Learning scenarios.
    `Stream`:

    - manages the "epoch" and "iteration" variables
    - allows to dump and restore loop's state: epoch, iteration, etc.
    - allows to customize the size of epoch
    - allows to change the underlying data loader on the fly
    - enables useful patterns

    .. rubric:: Tutorial

    Let's start with the most common training loop::

        loader = DataLoader(...)
        iteration = 0
        for epoch in range(max_epoch):
            for batch in loader:
                iteration += 1
                print('Epoch:', epoch, 'Iteration:', iteration)
                ...

    Let's enhance the loop using `Stream`::

        stream = delu.Stream(DataLoader(...))  # (A)
        for epoch in stream.epochs(max_epoch):  # (B)
            for batch in epoch:  # (C)
                print('Epoch:', stream.epoch, 'Iteration:', stream.iteration)  # (D)
                ...

    Some comments for the above code:

    - :code:`(A)` `Stream` is created by passing a dataloader as a single argument (in fact, you can pass any iterable object);
      the dataloader is accessible via `Stream.loader`
    - :code:`(B)` :code:`epoch` is an iterator over batches for one epoch
    - :code:`(C)` a progress bar for batches is displayed (for the whole training loop, not just for one epoch)
    - :code:`(D)` `Stream.epoch` and `Stream.iteration` are managed automatically

    Saving the loop's state and resuming the loop is possible with the methods
    `Stream.state_dict`, `Stream.load_state_dict`. In practice, it can look like this::

        model = ...
        optimizer = ...
        stream = delu.Stream(DataLoader(...))
        if load_from_checkpoint:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            stream.load_state_dict(checkpoint['stream'])
        ...
        for epoch in stream.epochs(...):
            for batch in epoch:
                ...
            torch.save(
                {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'stream': stream.state_dict(),
                },
                f'checkpoint_{stream.epoch}.pt'
            )

    Note:
        Stream's state does not include the loader's state. See `Stream.state_dict` and
        `Stream.load_state_dict` for details.

    In order to customize the epoch size, pass the size as the second argument::

        for epoch in stream.epochs(max_epoch, custom_epoch_size):
            for batch in epoch:
                ...

    Changing the underlying loader on the fly is possible at *any* moment (even in the
    middle of epoch) via `Stream.set_loader`. For example::

        for epoch in stream.epochs(max_epoch, custom_epoch_size):
            for batch in epoch:
                ...
                if need_new_data():
                    stream.set_loader(new_loader)

    If the method `Stream.epochs` does not fit your workflow and you want more control
    over the loop, there are more "low-level" methods (in fact, `Stream.epochs` is just
    a thin wrapper around them):

    - `Stream.increment_epoch`
    - `Stream.data`
    - `Stream.next`

    For example, the most common training loop can be implemented as follows::

        # A
        while stream.epoch < max_epoch:
            stream.increment_epoch()
            for batch in stream.data():
                ...

        # B
        while stream.epoch < max_epoch:
            stream.increment_epoch()
            for _ in range(len(stream.loader)):
                batch = stream.next()  # stream.iteration is incremented automatically
                ...

    The "infinite" stream of data can be implemented as follows::

        for item in stream.data(float('inf')):
            ...
            if condition:  # for example: `if stream.iteration % frequency == 0`
                ...

    Note:
        For better technical understanding, keep in mind that `Stream` simply
        encapsulates an "infinite iterator" that is constantly moving forward. The
        behavior is absolutely the same for both finite and infinite iterables and can
        be expressed with the following loop::

            while True:
                for item in loader:  # the loader which is passed to the constructor
                    ...

        Documentation for `Stream.next` and `Stream.data` provide helpful examples.
    """  # noqa: E501

    class _EpochData:
        def __init__(self, stream, size):
            self._stream = stream
            self._size = size
            self._start = self._stream.iteration

        def __iter__(self):
            return self

        def __next__(self):
            if (
                self._size is not None
                and self._stream.iteration - self._start >= self._size
            ):
                raise StopIteration()
            return self._stream.next()

    def __init__(self, loader: Iterable) -> None:
        """
        Args:
            loader: any kind of iterable (DataLoader, list, iterator, generator, ...)

        Examples:
            .. testcode::

                stream = delu.Stream([0, 1, 2, 3])
                stream = delu.Stream(range(10))
                import itertools
                stream = delu.Stream(itertools.repeat(0))

                from torch.utils.data import DataLoader, TensorDataset
                dataset = TensorDataset(torch.randn(10, 2))
                stream = delu.Stream(DataLoader(dataset, batch_size=3, shuffle=True))
        """
        assert _try_len(loader) != 0
        self._iteration = 0
        self._epoch = 0
        self._loader = loader
        self._iter: Optional[Iterator] = None
        self._progress_bar = None
        self._should_update_progress_bar = False

    @property
    def iteration(self) -> int:
        """Current iteration.

        Technically, the number of `Stream.next` calls.
        """
        return self._iteration

    @property
    def epoch(self) -> int:
        """Current epoch.

        Technically, the number of `Stream.increment_epoch` calls.
        """
        return self._epoch

    @property
    def loader(self) -> Iterable:
        """The underlying loader."""
        return self._loader

    def set_loader(self, loader: Iterable) -> None:
        """Set new loader.

        Args:
            loader:

        Examples:
            .. testcode::

                from itertools import repeat
                stream = delu.Stream(repeat(0))
                for x in stream.data(5):
                    print(stream.iteration, x)
                    if stream.iteration == 2:
                        stream.set_loader(repeat(1))

            .. testoutput::

                1 0
                2 0
                3 1
                4 1
                5 1
        """
        assert _try_len(loader) != 0
        self._loader = loader
        if self._iter is not None:
            self._iter = iter(loader)

    def _increment_iteration(self):
        self._iteration += 1

    def increment_epoch(self) -> None:
        """Increment `Stream.epoch`.

        Examples:
            .. testcode::

                stream = delu.Stream(range(5))
                assert stream.epoch == 0
                stream.increment_epoch()
                assert stream.epoch == 1
                stream.increment_epoch()
                assert stream.epoch == 2
        """
        self._epoch += 1

    def reload_iterator(self) -> None:
        """Set the underlying iterator to `iter(self.loader)`.

        If the underlying loader is a finite iterable, the method can be used to
        interrupt and skip the current epoch (i.e. skip its data). If the loader is an
        iterator, the method does nothing.

        Examples:
            .. testcode::

                stream = delu.Stream(range(5))
                assert stream.next() == 0
                assert stream.next() == 1
                stream.reload_iterator()
                assert stream.next() == 0

                stream = delu.Stream(iter(range(5)))
                assert stream.next() == 0
                assert stream.next() == 1
                stream.reload_iterator()
                assert stream.next() == 2
        """
        self._iter = iter(self.loader)

    def next(self) -> Any:
        """Get the next item and increment iteration.

        Returns:
            The next item.

        Examples:
            .. testcode::

                stream = delu.Stream(range(3))
                assert stream.iteration == 0
                assert stream.next() == 0
                assert stream.iteration == 1
                assert stream.next() == 1
                assert stream.next() == 2
                assert stream.next() == 0
                assert stream.iteration == 4

            .. code-block::

                while True:
                    x = stream.next()
                    ...
                    if stream.iteration % frequency:
                        ...
        """
        if self._iter is None:
            self._iter = iter(self._loader)
        if self._progress_bar is not None:
            if self._should_update_progress_bar:
                self._progress_bar.update()
            self._should_update_progress_bar = True
        try:
            value = next(self._iter)
        except StopIteration:
            self.reload_iterator()
            # If the following line raises StopIteration too, then the data is over
            # and the exception should be just propagated.
            value = next(self._iter)
        self._increment_iteration()
        return value

    def data(self, n_items: Optional[Union[int, float]] = None) -> Iterator:
        """Iterate over the loader.

        Under the hood, `Stream.next` is called, hence, `Stream.iteration` changes
        during iterations.

        Args:
            n_items: how many items to produce. If `None`, interpreted as
                :code:`len(self.loader)`. If `float`, must be :code:`float('inf')` or
                `math.inf`.

        Examples:
            .. testcode::

                stream = delu.Stream(range(5))
                assert list(stream.data()) == [0, 1, 2, 3, 4]
                assert list(stream.data(3)) == [0, 1, 2]
                # stream doesn't "start over"!
                assert list(stream.data(3)) == [3, 4, 0]
                assert list(stream.data(1)) == [1]
                assert list(stream.data(2)) == [2, 3]

            .. code-block::

                for x in stream.data(float('inf')):
                    ...
                    if stream.iteration % frequency:
                        ...
        """
        if isinstance(n_items, float):
            assert math.isinf(n_items)
        if n_items is None:
            if not isinstance(self.loader, Sized):
                raise ValueError()
            n_items = len(self.loader)
        return Stream._EpochData(self, n_items)

    def epochs(
        self,
        max_epoch: Union[int, float],
        epoch_size: Optional[Union[int, float]] = None,
        progress_bar_config: Optional[Dict[str, Any]] = _DEFAULT_PROGRESS_BAR_CONFIG,
    ) -> Iterator[Iterator[Any]]:
        """Iterate over data epochs.

        A shortcut for what is probably the most popular form of a training loop in Deep
        Learning (plus a progress bar)::

            for epoch in stream.epochs(max_epoch, epoch_size):
                for batch in epoch:
                    ...

            # is equivalent to:

            while stream.epoch < max_epoch:
                stream.increment_epoch()
                for batch in stream.data(epoch_size):
                    ...

        Args:
            max_epoch: defines the number of epochs. The loop keeps running while
                :code:`self.epoch < max_epoch`. If `float`, must be :code:`float('inf')`
                or `math.inf`.
            epoch_size: the number of data items in one epoch
                (is forwarded to `Stream.data`).
            progress_bar_config: if not `None` (the default value is :code:`{}`!), a
                progress bar for iterations will be displayed and the argument will be
                interpreted as key-word arguments for
                `tqdm <https://tqdm.github.io/docs/tqdm/#__init__>`_. The following
                key-word arguments will be automatically added if not presented in
                :code:`progress_bar_config`: :code:`initial`, :code:`total` (if can be
                inferred from the arguments and/or from `Stream.loader`). If
                [`ipywidgets`](https://ipywidgets.readthedocs.io) is installed and the
                program is running in JupyterLab (Jupyter Notebook), then the pretty
                widget is used instead of the text-based one.

        Returns:
            Iterator over iterators over data from `Stream.loader`.

        Examples:
            .. testcode::

                stream = delu.Stream(range(3))
                for epoch in stream.epochs(2):
                    for x in epoch:
                        print(x)
                    print('-')

            .. testoutput::

                0
                1
                2
                -
                0
                1
                2
                -

            .. testcode::

                stream = delu.Stream(range(3))
                for epoch in stream.epochs(3, 2):
                    for x in epoch:
                        print(x)
                    print('-')

            .. testoutput::

                0
                1
                -
                2
                0
                -
                1
                2
                -
        """
        if isinstance(max_epoch, float):
            assert math.isinf(max_epoch)
        progress_bar_config = None
        if progress_bar_config is not None:
            raise RuntimeError('unreachable')
            if progress_bar_config is _DEFAULT_PROGRESS_BAR_CONFIG:
                # the default value is MUTABLE, so let's deepcopy it
                # just in case it is modified later
                progress_bar_config = deepcopy(progress_bar_config)
            pbar_epoch_size = (
                _try_len(self.loader) if epoch_size is None else epoch_size
            )
            all_progress_bar_options = {
                'initial': self.iteration,
                'total': (
                    None
                    if (pbar_epoch_size is None or math.isinf(max_epoch))
                    else max_epoch * pbar_epoch_size
                ),
            }
            all_progress_bar_options.update(progress_bar_config)
            self._progress_bar = tqdm(**all_progress_bar_options)  # type: ignore  # noqa: F821
            self._should_update_progress_bar = False
        while self.epoch < max_epoch:
            self.increment_epoch()
            yield self.data(epoch_size)
        if self._should_update_progress_bar:
            assert self._progress_bar is not None
            self._progress_bar.update()

    def state_dict(self) -> Dict[str, Any]:
        """Get the stream's state.

        The result can be passed to `Stream.load_state_dict`. The result includes:

        - epoch
        - iteration

        Note:
            Fields related to data (loader, iterator etc.) are **NOT** included in the
            state. If you want to save the "state of data stream" then you have to save
            the state of corresponding random number generators separately.

        Returns:
            state

        Examples:
            .. testcode::

                stream = delu.Stream(range(10))
                assert stream.state_dict() == {'epoch': 0, 'iteration': 0}
                stream.next()
                stream.next()
                stream.increment_epoch()
                assert stream.state_dict() == {'epoch': 1, 'iteration': 2}
        """
        return {'iteration': self.iteration, 'epoch': self.epoch}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dictionary.

        Args:
            state_dict: state. Must be produced by `Stream.state_dict`.

        Note:
            The method does not affect data that is produced by `Stream.epochs`,
            `Stream.data`, `Stream.next` (see the examples below), i.e. the method
            only sets some "metadata" such as epoch, iteration etc. If you want to
            load the "state of data stream", you have to load the state of corresponding
            random number generators separately.

        Examples:

            .. testcode::

                stream = delu.Stream(range(10))
                stream.next()
                stream.increment_epoch()
                assert stream.state_dict() == {'epoch': 1, 'iteration': 1}

                new_stream = delu.Stream(range(10))
                new_stream.load_state_dict(stream.state_dict())
                assert new_stream.state_dict() == {'epoch': 1, 'iteration': 1}
                assert new_stream.next() == 0
                assert new_stream.state_dict() == {'epoch': 1, 'iteration': 2}
        """
        self._iteration = state_dict['iteration']
        self._epoch = state_dict['epoch']
