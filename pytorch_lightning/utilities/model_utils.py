from warnings import warn

warn(
    "`model_utils` package has been renamed to `model_helpers` since v1.2 and will be removed in v1.4",
    DeprecationWarning
)

from pytorch_lightning.utilities.model_helpers import *  # noqa: F403 E402 F401
