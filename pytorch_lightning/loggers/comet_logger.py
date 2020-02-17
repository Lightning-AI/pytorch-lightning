"""
.. warning:: `comet_logger` module has been renamed to `comet` since v0.6.0.
 The deprecated module name will be removed in v0.8.0.
"""

import warnings

warnings.warn("`comet_logger` module has been renamed to `comet` since v0.6.0."
              " The deprecated module name will be removed in v0.8.0.", DeprecationWarning)

from pytorch_lightning.loggers.comet import CometLogger  # noqa: E402
