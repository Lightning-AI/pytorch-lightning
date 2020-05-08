import os
import time
from pathlib import Path

import numpy as np
import torch
import pytest

from pytorch_lightning.profiler import AdvancedProfiler, SimpleProfiler, AutogradProfiler

PROFILER_OVERHEAD_MAX_TOLERANCE = 0.0005


def _get_python_cprofile_total_duration(profile):
    return sum([x.inlinetime for x in profile.getstats()])


def _get_pytorch_profiler_total_duration(events):
    total_time = sum([e.cpu_time + e.cuda_time for e in events])
    return total_time / 1e6  # convert microseconds to seconds


def _sleep_generator(durations):
    """
    the profile_iterable method needs an iterable in which we can ensure that we're
    properly timing how long it takes to call __next__
    """
    for duration in durations:
        time.sleep(duration)
        yield duration


@pytest.fixture
def simple_profiler():
    profiler = SimpleProfiler()
    return profiler


@pytest.fixture
def advanced_profiler(tmpdir):
    profiler = AdvancedProfiler(output_filename=os.path.join(tmpdir, "profiler.txt"))
    return profiler


@pytest.fixture
def autograd_profiler(tmpdir):
    profiler = AutogradProfiler(output_filename=os.path.join(tmpdir, "profiler.txt"))
    return profiler


# =====================
# Simple Profiler
# =====================
@pytest.mark.parametrize(["action", "expected"], [
    pytest.param("a", [3, 1]),
    pytest.param("b", [2]),
    pytest.param("c", [1])
])
def test_simple_profiler_durations(simple_profiler, action, expected):
    """Ensure the reported durations are reasonably accurate."""

    for duration in expected:
        with simple_profiler.profile(action):
            time.sleep(duration)

    # different environments have different precision when it comes to time.sleep()
    # see: https://github.com/PyTorchLightning/pytorch-lightning/issues/796
    np.testing.assert_allclose(
        simple_profiler.recorded_durations[action], expected, rtol=0.2
    )


@pytest.mark.parametrize(["action", "expected"], [
    pytest.param("a", [3, 1]),
    pytest.param("b", [2]),
    pytest.param("c", [1])
])
def test_simple_profiler_iterable_durations(simple_profiler, action, expected):
    """Ensure the reported durations are reasonably accurate."""
    iterable = _sleep_generator(expected)

    for _ in simple_profiler.profile_iterable(iterable, action):
        pass

    # we exclude the last item in the recorded durations since that's when StopIteration is raised
    np.testing.assert_allclose(
        simple_profiler.recorded_durations[action][:-1], expected, rtol=0.2
    )


def test_simple_profiler_overhead(simple_profiler, n_iter=5):
    """Ensure that the profiler doesn't introduce too much overhead during training."""
    for _ in range(n_iter):
        with simple_profiler.profile("no-op"):
            pass

    durations = np.array(simple_profiler.recorded_durations["no-op"])
    assert all(durations < PROFILER_OVERHEAD_MAX_TOLERANCE)


def test_simple_profiler_describe(caplog, simple_profiler):
    """Ensure the profiler won't fail when reporting the summary."""
    simple_profiler.describe()

    assert "Profiler Report" in caplog.text


def test_simple_profiler_value_errors(simple_profiler):
    """Ensure errors are raised where expected."""

    action = "test"
    with pytest.raises(ValueError):
        simple_profiler.stop(action)

    simple_profiler.start(action)

    with pytest.raises(ValueError):
        simple_profiler.start(action)

    simple_profiler.stop(action)


# =====================
# Advanced Profiler
# =====================
@pytest.mark.parametrize(["action", "expected"], [
    pytest.param("a", [3, 1]),
    pytest.param("b", [2]),
    pytest.param("c", [1])
])
def test_advanced_profiler_durations(advanced_profiler, action, expected):
    """Ensure the reported durations are reasonably accurate."""

    for duration in expected:
        with advanced_profiler.profile(action):
            time.sleep(duration)

    # different environments have different precision when it comes to time.sleep()
    # see: https://github.com/PyTorchLightning/pytorch-lightning/issues/796
    recored_total_duration = _get_python_cprofile_total_duration(
        advanced_profiler.profiled_actions[action]
    )
    expected_total_duration = np.sum(expected)
    np.testing.assert_allclose(
        recored_total_duration, expected_total_duration, rtol=0.2
    )


@pytest.mark.parametrize(["action", "expected"], [
    pytest.param("a", [3, 1]),
    pytest.param("b", [2]),
    pytest.param("c", [1])
])
def test_advanced_profiler_iterable_durations(advanced_profiler, action, expected):
    """Ensure the reported durations are reasonably accurate."""
    iterable = _sleep_generator(expected)

    for _ in advanced_profiler.profile_iterable(iterable, action):
        pass

    recored_total_duration = _get_python_cprofile_total_duration(
        advanced_profiler.profiled_actions[action]
    )
    expected_total_duration = np.sum(expected)
    np.testing.assert_allclose(
        recored_total_duration, expected_total_duration, rtol=0.2
    )


def test_advanced_profiler_overhead(advanced_profiler, n_iter=5):
    """Ensure that the profiler doesn't introduce too much overhead during training."""
    for _ in range(n_iter):
        with advanced_profiler.profile("no-op"):
            pass

    action_profile = advanced_profiler.profiled_actions["no-op"]
    total_duration = _get_python_cprofile_total_duration(action_profile)
    average_duration = total_duration / n_iter
    assert average_duration < PROFILER_OVERHEAD_MAX_TOLERANCE


def test_advanced_profiler_describe(tmpdir, advanced_profiler):
    """Ensure the profiler won't fail when reporting the summary."""
    # record at least one event
    with advanced_profiler.profile("test"):
        pass
    # log to stdout and print to file
    advanced_profiler.describe()
    data = Path(advanced_profiler.output_fname).read_text()
    assert len(data) > 0


def test_advanced_profiler_value_errors(advanced_profiler):
    """Ensure errors are raised where expected."""

    action = "test"
    with pytest.raises(ValueError):
        advanced_profiler.stop(action)

    advanced_profiler.start(action)
    advanced_profiler.stop(action)


# =====================
# Autograd Profiler
# =====================

def test_autograd_profiler_overhead(autograd_profiler, n_iter=5):
    """Ensure that the profiler doesn't introduce too much overhead during training."""
    for _ in range(n_iter):
        with autograd_profiler.profile("no-op"):
            a = torch.ones(42)
            b = torch.abs(a)
            c = a + b

    action_profile = autograd_profiler.profiled_actions["no-op"]
    total_duration = _get_pytorch_profiler_total_duration(action_profile)
    average_duration = total_duration / n_iter
    assert average_duration < PROFILER_OVERHEAD_MAX_TOLERANCE


def test_autograd_profiler_describe(tmpdir, autograd_profiler):
    """Ensure the profiler won't fail when reporting the summary."""
    with autograd_profiler.profile("test"):
        pass

    # log to stdout and print to file
    autograd_profiler.describe()
    data = Path(autograd_profiler.output_fname).read_text()
    assert len(data) > 0


def test_autograd_profiler_value_errors(autograd_profiler):
    """Ensure errors are raised where expected."""

    action = "test"
    with pytest.raises(ValueError):
        autograd_profiler.stop(action)

    autograd_profiler.start(action)
    autograd_profiler.stop(action)
