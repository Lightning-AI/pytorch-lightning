"""
.. warning:: `root_module.model_saving` module has been renamed to `core.saving` since v0.6.0.
 The deprecated module name will be removed in v0.8.0.
"""

import warnings

warnings.warn("`root_module.model_saving` module has been renamed to `core.saving` since v0.6.0."
              " The deprecated module name will be removed in v0.8.0.", DeprecationWarning)

from pytorch_lightning.core.saving import *  # noqa: E403
