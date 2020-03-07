"""
.. warning:: `root_module` module has been renamed to `lightning` since v0.6.0.
 The deprecated module name will be removed in v0.8.0.
"""

import warnings

from pytorch_lightning.core.lightning import *  # noqa: F403

warnings.warn("`root_module` module has been renamed to `lightning` since v0.6.0."
              " The deprecated module name will be removed in v0.8.0.", DeprecationWarning)
