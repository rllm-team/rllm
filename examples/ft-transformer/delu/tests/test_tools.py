import pickle
from time import perf_counter, sleep

import pytest

import delu
import delu.tools


def test_early_stopping():
    with pytest.raises(ValueError):
        delu.tools.EarlyStopping(0, mode='max')
    with pytest.raises(ValueError):
        delu.tools.EarlyStopping(1, mode='hello')
    with pytest.raises(ValueError):
        delu.tools.EarlyStopping(1, mode='min', min_delta=-1.0)

    es = delu.tools.EarlyStopping(1, mode='min')
    es.update(1.0)
    assert not es.should_stop()
    es.update(1.0)
    assert es.should_stop()

    es.reset_unsuccessful_updates()
    es.update(1.0)
    assert es.should_stop()

    es.reset()
    es.update(1.0)
    assert not es.should_stop()
    es.update(1.0)
    assert es.should_stop()

    for mode in ['min', 'max']:
        sign = -1.0 if mode == 'min' else 1.0

        es = delu.tools.EarlyStopping(2, mode=mode)
        es.update(0.0)
        assert not es.should_stop()
        es.update(sign * 1.0)
        assert not es.should_stop()
        es.update(sign * 1.0)
        assert not es.should_stop()
        es.update(sign * 1.0)
        assert es.should_stop()
        es.update(sign * 2.0)
        assert not es.should_stop()
        es.update(sign * 2.0)
        assert not es.should_stop()
        es.update(sign * 2.0)
        assert es.should_stop()

        min_delta = 0.1
        es = delu.tools.EarlyStopping(1, mode=mode, min_delta=min_delta)
        es.update(0.0)
        assert not es.should_stop()
        es.update(sign * 2 * min_delta)
        assert not es.should_stop()
        es.update(sign * 2.99 * min_delta)
        assert es.should_stop()


def test_timer():
    delu.tools.Timer().pause()

    # initial state, run
    timer = delu.tools.Timer()
    sleep(0.001)
    assert not timer.elapsed()
    timer.run()
    assert timer.elapsed()

    # pause
    timer.pause()
    timer.pause()  # two pauses in a row
    x = timer.elapsed()
    sleep(0.001)
    assert timer.elapsed() == x

    # run
    timer.pause()
    x = timer.elapsed()
    timer.run()
    timer.run()  # two runs in a row
    assert timer.elapsed() != x
    timer.pause()
    x = timer.elapsed()
    sleep(0.001)
    assert timer.elapsed() == x
    timer.run()

    # reset
    timer.reset()
    assert not timer.elapsed()


def test_timer_measurements():
    x = perf_counter()
    sleep(0.1)
    correct = perf_counter() - x
    timer = delu.tools.Timer()
    timer.run()
    sleep(0.1)
    actual = timer.elapsed()
    # the allowed deviation was obtained from manual runs on my laptop so the test may
    # behave differently on other hardware
    assert actual == pytest.approx(correct, abs=0.01)


def test_timer_context():
    with delu.tools.Timer() as timer:
        sleep(0.01)
    assert timer.elapsed() > 0.01
    assert timer.elapsed() == timer.elapsed()

    timer = delu.tools.Timer()
    timer.run()
    sleep(0.01)
    timer.pause()
    with timer:
        sleep(0.01)
    assert timer.elapsed() > 0.02
    assert timer.elapsed() == timer.elapsed()


def test_timer_pickle():
    timer = delu.tools.Timer()
    timer.run()
    sleep(0.01)
    timer.pause()
    value = timer.elapsed()
    sleep(0.01)
    assert pickle.loads(pickle.dumps(timer)).elapsed() == timer.elapsed() == value


def test_timer_format():
    def make_timer(x):
        timer = delu.tools.Timer()
        timer._shift = x
        return timer

    assert str(make_timer(1)) == '0:00:01'
    assert format(make_timer(1.1)) == '0:00:01.100000'
    assert f'{make_timer(7321):%Hh %Mm %Ss}' == '02h 02m 01s'


def test_progress_tracker():
    score = -999999999

    # test initial state
    tracker = delu.ProgressTracker(0)
    assert not tracker.success
    assert not tracker.fail

    # test successful update
    tracker.update(score)
    assert tracker.best_score == score
    assert tracker.success

    # test failed update
    tracker.update(score)
    assert tracker.best_score == score
    assert tracker.fail

    # test forget_bad_updates, reset
    tracker.forget_bad_updates()
    assert tracker.best_score == score
    tracker.reset()
    assert tracker.best_score is None
    assert not tracker.success and not tracker.fail

    # test positive patience
    tracker = delu.ProgressTracker(1)
    tracker.update(score - 1)
    assert tracker.success
    tracker.update(score)
    assert tracker.success
    tracker.update(score)
    assert not tracker.success and not tracker.fail
    tracker.update(score)
    assert tracker.fail

    # test positive min_delta
    tracker = delu.ProgressTracker(0, 2)
    tracker.update(score - 2)
    assert tracker.success
    tracker.update(score)
    assert tracker.fail
    tracker.reset()
    tracker.update(score - 3)
    tracker.update(score)
    assert tracker.success

    # patience=None
    tracker = delu.ProgressTracker(None)
    for i in range(100):
        tracker.update(-i)
        assert not tracker.fail
