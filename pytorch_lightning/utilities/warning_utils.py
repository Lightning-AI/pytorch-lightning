from pytorch_lightning.utilities import rank_zero_deprecation

rank_zero_deprecation("`warning_utils` package has been renamed to `warnings` since v1.2 and will be removed in v1.4")

from pytorch_lightning.utilities.warnings import *  # noqa: F403 E402 F401
