"""
.. warning:: `test_tube_logger` module has been renamed to `test_tube` since v0.6.0.
 The deprecated module name will be removed in v0.8.0.
"""

import warnings

warnings.warn("`test_tube_logger` module has been renamed to `test_tube` since v0.6.0."
              " The deprecated module name will be removed in v0.8.0.", DeprecationWarning)

from pytorch_lightning.loggers.test_tube import TestTubeLogger  # noqa: E402
