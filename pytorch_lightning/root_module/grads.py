"""
.. warning:: `root_module.grads` module has been renamed to `core.grads` since v0.6.0.
 The deprecated module name will be removed in v0.8.0.
"""

from pytorch_lightning.utilities import rank_zero_warn

rank_zero_warn("`root_module.grads` module has been renamed to `core.grads` since v0.6.0."
               " The deprecated module name will be removed in v0.8.0.", DeprecationWarning)

from pytorch_lightning.core.grads import *  # noqa: F403
