from pytorch_lightning.utilities import rank_zero_deprecation

rank_zero_deprecation("`argparse_utils` package has been renamed to `argparse` since v1.2 and will be removed in v1.4")

from pytorch_lightning.utilities.argparse import *  # noqa: F403 E402 F401
