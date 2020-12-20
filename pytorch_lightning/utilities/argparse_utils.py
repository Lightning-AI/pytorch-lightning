from warnings import warn

warn("`argparse_utils` package has been renamed to `argparse` since v1.2 and will be removed in v1.3",
     DeprecationWarning)

from pytorch_lightning.utilities.argparse import *  # noqa: F403
