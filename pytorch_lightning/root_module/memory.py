"""
.. warning:: `root_module.memory` module has been renamed to `core.memory` since v0.6.0.
 The deprecated module name will be removed in v0.8.0.
"""

from pytorch_lightning.utilities import rank_zero_warn

rank_zero_warn("`root_module.memory` module has been renamed to `core.memory` since v0.6.0."
               " The deprecated module name will be removed in v0.8.0.", DeprecationWarning)

from pytorch_lightning.core.memory import *  # noqa: F403 E402
