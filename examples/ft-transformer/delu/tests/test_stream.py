import itertools
import math

from pytest import mark, raises

from delu import Stream

from .util import ignore_deprecated_warning


@ignore_deprecated_warning
def test_properties():
    loader = [0]
    stream = Stream(loader)
    assert stream.iteration == 0
    assert stream.epoch == 0
    assert stream.loader is loader
    for attr in 'iteration', 'epoch', 'loader':
        with raises(AttributeError):
            setattr(stream, attr, None)


@ignore_deprecated_warning
def test_bad_loader():
    with raises(AssertionError):
        Stream([])


@ignore_deprecated_warning
def test_increment_epoch():
    stream = Stream(range(3))
    for i in range(10):
        assert stream.epoch == i
        stream.increment_epoch()


@ignore_deprecated_warning
def test_set_loader():
    a = itertools.repeat(0)
    b = itertools.repeat(1)
    stream = Stream([2])
    stream.set_loader(a)
    for x in range(10):
        assert stream.next() == next(b if x % 2 else a)
        stream.set_loader(a if x % 2 else b)


@ignore_deprecated_warning
def test_reload_iterator():
    n = 10
    stream = Stream(range(n))
    for x in range(n):
        assert stream.next() == 0
        stream.reload_iterator()

    stream = Stream(iter(range(n)))
    for x in range(n):
        assert stream.next() == x
        stream.reload_iterator()
    with raises(StopIteration):
        stream.next()


@ignore_deprecated_warning
@mark.parametrize('n', range(1, 5))
def test_next_iteration(n):
    stream = Stream(range(n))
    assert stream.iteration == 0

    for epoch in range(2):
        for x in range(n):
            assert stream.next() == x
            assert stream.iteration == x + epoch * n + 1

    stream = Stream(iter(range(n)))
    for _ in range(n):
        stream.next()
    with raises(StopIteration):
        stream.next()
    with raises(StopIteration):
        stream.next()


@ignore_deprecated_warning
@mark.parametrize('n', range(1, 6))
def test_data(n):
    # iterables that are not iterators
    n_epochs = 10
    stream = Stream(range(n))
    assert [list(stream.data(0)) for _ in range(n_epochs)] == [[]] * n_epochs

    for epoch_size in [None, *list(range(1, 2 * n))]:
        effective_epoch_size = n if epoch_size is None else epoch_size
        max_iteration = n_epochs * effective_epoch_size
        stream = Stream(range(n))
        actual = []
        while stream.iteration < max_iteration:
            actual.append(list(stream.data(epoch_size)))
        assert stream.iteration == max_iteration
        flat_correct = [x % n for x in range(max_iteration)]
        correct = [
            flat_correct[i * effective_epoch_size : (i + 1) * effective_epoch_size]
            for i in range(n_epochs)
        ]
        assert actual == correct

    # count=inf
    stream = Stream(range(n))
    assert [x for _, x in zip(range(1000), stream.data(math.inf))] == [
        x % n for x in range(1000)
    ]

    # infinite iterators
    stream = Stream(itertools.count())
    with raises(ValueError):
        stream.data()
    n_epochs = 10
    for epoch_size in range(1, n):
        stream = Stream(itertools.count())
        actual = [list(stream.data(epoch_size)) for _ in range(n_epochs)]
        flat_correct = list(range(n_epochs * epoch_size))
        correct = [
            flat_correct[i * epoch_size : (i + 1) * epoch_size] for i in range(n_epochs)
        ]
        assert actual == correct

    # finite iterators
    stream = Stream(iter(range(n)))
    with raises(ValueError):
        stream.data()
    n_epochs = 10
    for epoch_size in range(1, 2 * n):
        stream = Stream(iter(range(n)))
        actual = []
        for _ in range(n_epochs):
            actual.append(list(stream.data(epoch_size)))
        flat_correct = list(range(n))
        correct = [
            flat_correct[i * epoch_size : (i + 1) * epoch_size] for i in range(n_epochs)
        ]
        assert actual == correct

    # bad input
    with raises(AssertionError):
        stream.data(10.0)


@ignore_deprecated_warning
def test_epochs():
    stream = Stream(range(3))
    with raises(AssertionError):
        next(stream.epochs(1.0, progress_bar_config=None))
    correct = [0, 1, 2]
    for epoch in stream.epochs(2, progress_bar_config=None):
        assert list(epoch) == correct
    correct = [[0, 1], [2, 0], [1, 2]]
    for i, epoch in enumerate(stream.epochs(2, progress_bar_config=None)):
        assert list(epoch) == correct[i]
    for i, epoch in zip(
        enumerate(stream.epochs(math.inf, progress_bar_config=None)), range(1000)
    ):
        pass
    for (i, epoch), _ in zip(enumerate(stream.epochs(math.inf, math.inf)), range(10)):
        for (j, _), _ in zip(enumerate(epoch), range(10)):
            pass
        assert j == 9
    assert i == 9


@ignore_deprecated_warning
def test_state_dict():
    stream = Stream(range(10))
    stream.next()
    stream.increment_epoch()
    assert stream.state_dict() == {'epoch': 1, 'iteration': 1}

    new_stream = Stream(range(10))
    new_stream.load_state_dict(stream.state_dict())
    assert new_stream.state_dict() == {'epoch': 1, 'iteration': 1}
    assert new_stream.next() == 0
    assert new_stream.state_dict() == {'epoch': 1, 'iteration': 2}
    assert new_stream.next() == 1

    assert stream.next() == 1
