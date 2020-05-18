"""
.. warning:: `root_module.decorators` module has been renamed to `core.decorators` since v0.6.0.
 The deprecated module name will be removed in v0.8.0.
"""

from pytorch_lightning.utilities import rank_zero_warn

rank_zero_warn("`root_module.decorators` module has been renamed to `core.decorators` since v0.6.0."
               " The deprecated module name will be removed in v0.8.0.", DeprecationWarning)

from pytorch_lightning.core.decorators import *  # noqa: F403 E402
