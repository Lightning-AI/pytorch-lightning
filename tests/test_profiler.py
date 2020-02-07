import time

import numpy as np

from pytorch_lightning.profiler import Profiler, AdvancedProfiler


def _assert_almost_equal(got, expect, tol=0.1):
    expect = np.array(expect)
    got = np.array(got)
    is_any_lower = np.any(np.less(got, expect - tol))
    is_any_above = np.any(np.greater(got, expect + tol))
    if is_any_lower or is_any_above:
        raise AssertionError("Expected: %s\nGot: %s\n which is out of the range +/- %f",
                             expect, got, tol)


def test_simple_profiler():
    p = Profiler()

    with p.profile("a"):
        time.sleep(3)

    with p.profile("a"):
        time.sleep(1)

    with p.profile("b"):
        time.sleep(2)

    with p.profile("c"):
        time.sleep(1)

    # different environments have different precision when it comes to time.sleep()
    # see: https://github.com/PyTorchLightning/pytorch-lightning/issues/796
    _assert_almost_equal(p.recorded_durations["a"], [3, 1], tol=0.1)
    _assert_almost_equal(p.recorded_durations["b"], [2], tol=0.1)
    _assert_almost_equal(p.recorded_durations["c"], [1], tol=0.1)


def test_advanced_profiler():
    def _get_duration(profile):
        return sum([x.totaltime for x in profile.getstats()])

    p = AdvancedProfiler()

    with p.profile("a"):
        time.sleep(3)

    with p.profile("a"):
        time.sleep(1)

    with p.profile("b"):
        time.sleep(2)

    with p.profile("c"):
        time.sleep(1)

    # different environments have different precision when it comes to time.sleep()
    # see: https://github.com/PyTorchLightning/pytorch-lightning/issues/796
    a_duration = _get_duration(p.profiled_actions["a"])
    _assert_almost_equal(a_duration, [4], tol=0.5)
    b_duration = _get_duration(p.profiled_actions["b"])
    _assert_almost_equal(b_duration, [2], tol=0.5)
    c_duration = _get_duration(p.profiled_actions["c"])
    _assert_almost_equal(c_duration, [1], tol=0.5)
