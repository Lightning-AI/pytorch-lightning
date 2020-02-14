import time

import numpy as np

from pytorch_lightning.profiler import Profiler, AdvancedProfiler


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
    np.testing.assert_allclose(p.recorded_durations["a"], [3, 1], rtol=0.2)
    np.testing.assert_allclose(p.recorded_durations["b"], [2], rtol=0.2)
    np.testing.assert_allclose(p.recorded_durations["c"], [1], rtol=0.2)

    # ensure this doesn't throw any errors
    description = p.describe()


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
    np.testing.assert_allclose(a_duration, [4], rtol=0.2)
    b_duration = _get_duration(p.profiled_actions["b"])
    np.testing.assert_allclose(b_duration, [2], rtol=0.2)
    c_duration = _get_duration(p.profiled_actions["c"])
    np.testing.assert_allclose(c_duration, [1], rtol=0.2)

    # ensure this doesn't throw any errors
    description = p.describe()
