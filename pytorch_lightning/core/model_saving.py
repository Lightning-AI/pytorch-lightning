import warnings

warnings.warn("`model_saving` module has been renamed to `saving` since v0.5.3"
              " and will be removed in v0.8.0", DeprecationWarning)

from pytorch_lightning.core.saving import ModelIO  # noqa: E402
