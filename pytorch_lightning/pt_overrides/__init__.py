"""
.. warning:: `pt_overrides` package has been renamed to `overrides` since v0.6.0.
 The deprecated module name will be removed in v0.8.0.
"""

from pytorch_lightning.utilities import rank_zero_warn

rank_zero_warn("`pt_overrides` package has been renamed to `overrides` since v0.6.0."
               " The deprecated module name will be removed in v0.8.0.", DeprecationWarning)
