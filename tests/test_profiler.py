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
