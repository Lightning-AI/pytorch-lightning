import time

import numpy as np
import pytest

from pytorch_lightning.profiler import Profiler, AdvancedProfiler

PROFILER_OVERHEAD_MAX_TOLERANCE = 0.0001


@pytest.fixture
def simple_profiler():
    profiler = Profiler()
    return profiler


@pytest.fixture
def advanced_profiler():
    profiler = AdvancedProfiler()
    return profiler


@pytest.mark.parametrize("action,expected", [("a", [3, 1]), ("b", [2]), ("c", [1])])
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


def test_simple_profiler_overhead(simple_profiler, n_iter=5):
    """Ensure that the profiler doesn't introduce too much overhead during training."""
    for _ in range(n_iter):
        with simple_profiler.profile("no-op"):
            pass

    durations = np.array(simple_profiler.recorded_durations["no-op"])
    assert all(durations < PROFILER_OVERHEAD_MAX_TOLERANCE)


def test_simple_profiler_describe(simple_profiler):
    """Ensure the profiler won't fail when reporting the summary."""
    simple_profiler.describe()


@pytest.mark.parametrize("action,expected", [("a", [3, 1]), ("b", [2]), ("c", [1])])
def test_advanced_profiler_durations(advanced_profiler, action, expected):
    def _get_total_duration(profile):
        return sum([x.totaltime for x in profile.getstats()])

    for duration in expected:
        with advanced_profiler.profile(action):
            time.sleep(duration)

    # different environments have different precision when it comes to time.sleep()
    # see: https://github.com/PyTorchLightning/pytorch-lightning/issues/796
    recored_total_duration = _get_total_duration(
        advanced_profiler.profiled_actions[action]
    )
    expected_total_duration = np.sum(expected)
    np.testing.assert_allclose(
        recored_total_duration, expected_total_duration, rtol=0.2
    )


def test_advanced_profiler_overhead(advanced_profiler, n_iter=5):
    """
    ensure that the profiler doesn't introduce too much overhead during training
    """
    for _ in range(n_iter):
        with advanced_profiler.profile("no-op"):
            pass

    action_profile = advanced_profiler.profiled_actions["no-op"]
    total_duration = sum([x.totaltime for x in action_profile.getstats()])
    average_duration = total_duration / n_iter
    assert average_duration < PROFILER_OVERHEAD_MAX_TOLERANCE


def test_advanced_profiler_describe(advanced_profiler):
    """
    ensure the profiler won't fail when reporting the summary
    """
    advanced_profiler.describe()
