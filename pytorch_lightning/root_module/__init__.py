"""
.. warning:: `root_module` package has been renamed to `core` since v0.6.0.
 The deprecated package name will be removed in v0.8.0.
"""

from pytorch_lightning.utilities import rank_zero_warn

rank_zero_warn("`root_module` package has been renamed to `core` since v0.6.0."
               " The deprecated package name will be removed in v0.8.0.", DeprecationWarning)
