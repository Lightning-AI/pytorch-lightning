import warnings

warnings.warn("`mlflow_logger` module has been renamed to `mlflow` since v0.5.3"
              " and will be removed in v0.8.0", DeprecationWarning)

from pytorch_lightning.logging.mlflow import MLFlowLogger  # noqa: E402
