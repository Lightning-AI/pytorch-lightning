"""
.. warning:: `logging` package has been renamed to `loggers` since v0.7.0 and will be removed in v0.9.0
"""

from pytorch_lightning.utilities import rank_zero_warn

rank_zero_warn("`logging.test_tube` module has been renamed to `loggers.test_tube` since v0.7.0."
               " The deprecated module name will be removed in v0.9.0.", DeprecationWarning)

from pytorch_lightning.loggers.test_tube import TestTubeLogger  # noqa: F403
