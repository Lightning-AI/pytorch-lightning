"""
.. warning:: `root_module.memory` module has been renamed to `core.memory` since v0.6.0.
 The deprecated module name will be removed in v0.8.0.
"""

import warnings

warnings.warn("`root_module.memory` module has been renamed to `core.memory` since v0.6.0."
              " The deprecated module name will be removed in v0.8.0.", DeprecationWarning)

from pytorch_lightning.core.memory import *  # noqa: F403
