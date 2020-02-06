from pytorch_lightning.utilities.profiler import Profiler, AdvancedProfiler
import time
import numpy as np


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
    np.testing.assert_almost_equal(p.recorded_durations["a"], [3, 1], decimal=1)
    np.testing.assert_almost_equal(p.recorded_durations["b"], [2], decimal=1)
    np.testing.assert_almost_equal(p.recorded_durations["c"], [1], decimal=1)


def test_advanced_profiler():
    def get_duration(profile):
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

    a_duration = get_duration(p.profiled_actions["a"])
    np.testing.assert_almost_equal(a_duration, [4], decimal=1)
    b_duration = get_duration(p.profiled_actions["b"])
    np.testing.assert_almost_equal(b_duration, [2], decimal=1)
    c_duration = get_duration(p.profiled_actions["c"])
    np.testing.assert_almost_equal(c_duration, [1], decimal=1)
