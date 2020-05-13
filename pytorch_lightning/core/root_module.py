"""
.. warning:: `root_module` module has been renamed to `lightning` since v0.6.0.
 The deprecated module name will be removed in v0.8.0.
"""

from pytorch_lightning.utilities import rank_zero_warn

rank_zero_warn("`root_module` module has been renamed to `lightning` since v0.6.0."
               " The deprecated module name will be removed in v0.8.0.", DeprecationWarning)

from pytorch_lightning.core.lightning import *  # noqa: F403 E402
