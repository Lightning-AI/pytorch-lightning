"""
.. warning:: `root_module` module has been renamed to `lightning` since v0.6.0 and will be removed in v0.8.0
"""

import warnings

warnings.warn("`root_module` module has been renamed to `lightning` since v0.6.0"
              " and will be removed in v0.8.0", DeprecationWarning)

from pytorch_lightning.core.lightning import LightningModule  # noqa: E402
