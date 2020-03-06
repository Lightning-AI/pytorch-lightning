"""
.. warning:: `root_module.root_module` module has been renamed to `core.lightning` since v0.6.0.
 The deprecated module name will be removed in v0.8.0.
"""

import warnings

warnings.warn("`root_module.root_module` module has been renamed to `core.lightning` since v0.6.0."
              " The deprecated module name will be removed in v0.8.0.", DeprecationWarning)

from pytorch_lightning.core.lightning import *  # noqa: F403
