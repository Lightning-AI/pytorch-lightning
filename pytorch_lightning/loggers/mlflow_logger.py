"""
.. warning:: `mlflow_logger` module has been renamed to `mlflow` since v0.6.0.
 The deprecated module name will be removed in v0.8.0.
"""

import warnings

warnings.warn("`mlflow_logger` module has been renamed to `mlflow` since v0.6.0."
              " The deprecated module name will be removed in v0.8.0.", DeprecationWarning)

from pytorch_lightning.loggers.mlflow import MLFlowLogger  # noqa: E402
