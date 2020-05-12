"""
.. warning:: `logging` package has been renamed to `loggers` since v0.7.0.
 The deprecated package name will be removed in v0.9.0.
"""

from pytorch_lightning.utilities import rank_zero_warn

rank_zero_warn("`logging` package has been renamed to `loggers` since v0.7.0"
               " The deprecated package name will be removed in v0.9.0.", DeprecationWarning)

from pytorch_lightning.loggers import *  # noqa: F403 E402
