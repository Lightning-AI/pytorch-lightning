"""
.. warning:: `comet_logger` module has been renamed to `comet` since v0.6.0 and will be removed in v0.8.0
"""

import warnings

warnings.warn("`comet_logger` module has been renamed to `comet` since v0.6.0"
              " and will be removed in v0.8.0", DeprecationWarning)

from pytorch_lightning.logging.comet import CometLogger  # noqa: E402
