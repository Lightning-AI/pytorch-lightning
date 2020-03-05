"""
.. warning:: `logging` package has been renamed to `loggers` since v0.7.0.
 The deprecated package name will be removed in v0.9.0.
"""

import warnings

warnings.warn("`logging` package has been renamed to `loggers` since v0.7.0"
              " The deprecated package name will be removed in v0.9.0.", DeprecationWarning)

from pytorch_lightning.loggers import *  # noqa: F403
from pytorch_lightning.loggers import base, tensorboard  # noqa: F403
