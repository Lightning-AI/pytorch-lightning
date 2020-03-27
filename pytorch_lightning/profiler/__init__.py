"""
.. warning:: `profiler` package has been renamed to `profiling` since v0.7.2 and will be removed in v0.9.0
"""

import warnings

warnings.warn("`profiler` package has been renamed to `profiling` since v0.7.2."
              " The deprecated module name will be removed in v0.9.0.", DeprecationWarning)

from pytorch_lightning.profiling.profilers import (  # noqa: F403
    SimpleProfiler, AdvancedProfiler, BaseProfiler, PassThroughProfiler
)
