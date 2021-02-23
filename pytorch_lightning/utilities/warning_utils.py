from warnings import warn

warn(
    "`warning_utils` package has been renamed to `warnings` since v1.2 and will be removed in v1.4", DeprecationWarning
)

from pytorch_lightning.utilities.warnings import *  # noqa: F403 E402 F401
