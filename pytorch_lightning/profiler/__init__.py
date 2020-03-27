"""
.. warning:: `profiler` package has been renamed to `debugging` since v0.6.0.
 The deprecated module name will be removed in v0.8.0.
"""

import warnings

warnings.warn("`profiler` package has been renamed to `debugging` since v0.7.2."
              " The deprecated module name will be removed in v0.9.0.", DeprecationWarning)

from pytorch_lightning.debugging import (  # noqa: E402
    BaseProfiler, Profiler, AdvancedProfiler, PassThroughProfiler
)
