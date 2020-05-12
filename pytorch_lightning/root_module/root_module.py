"""
.. warning:: `root_module.root_module` module has been renamed to `core.lightning` since v0.6.0.
 The deprecated module name will be removed in v0.8.0.
"""

from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.core.lightning import *  # noqa: F403

rank_zero_warn("`root_module.root_module` module has been renamed to `core.lightning` since v0.6.0."
               " The deprecated module name will be removed in v0.8.0.", DeprecationWarning)
