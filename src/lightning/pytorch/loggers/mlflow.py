# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
MLflow Logger
-------------
"""

import os
from argparse import Namespace
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

from typing_extensions import override

from lightning.fabric.utilities.imports import _raise_enterprise_not_available
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers.logger import Logger, rank_zero_experiment
from lightning.pytorch.utilities.rank_zero import rank_zero_only

if TYPE_CHECKING:
    from mlflow.tracking import MlflowClient


class MLFlowLogger(Logger):
    """Log using `MLflow <https://mlflow.org>`_.

    Install it with pip:

    .. code-block:: bash

        pip install mlflow  # or mlflow-skinny

    .. code-block:: python

        from lightning.pytorch import Trainer
        from lightning.pytorch.loggers import MLFlowLogger

        mlf_logger = MLFlowLogger(experiment_name="lightning_logs", tracking_uri="file:./ml-runs")
        trainer = Trainer(logger=mlf_logger)

    Use the logger anywhere in your :class:`~lightning.pytorch.core.LightningModule` as follows:

    .. code-block:: python

        from lightning.pytorch import LightningModule


        class LitModel(LightningModule):
            def training_step(self, batch, batch_idx):
                # example
                self.logger.experiment.whatever_ml_flow_supports(...)

            def any_lightning_module_function_or_hook(self):
                self.logger.experiment.whatever_ml_flow_supports(...)

    Args:
        experiment_name: The name of the experiment.
        run_name: Name of the new run. The `run_name` is internally stored as a ``mlflow.runName`` tag.
            If the ``mlflow.runName`` tag has already been set in `tags`, the value is overridden by the `run_name`.
        tracking_uri: Address of local or remote tracking server.
            If not provided, defaults to `MLFLOW_TRACKING_URI` environment variable if set, otherwise it falls
            back to `file:<save_dir>`.
        tags: A dictionary tags for the experiment.
        save_dir: A path to a local directory where the MLflow runs get saved.
            Defaults to `./mlruns` if `tracking_uri` is not provided.
            Has no effect if `tracking_uri` is provided.
        log_model: Log checkpoints created by :class:`~lightning.pytorch.callbacks.model_checkpoint.ModelCheckpoint`
            as MLFlow artifacts.

            * if ``log_model == 'all'``, checkpoints are logged during training.
            * if ``log_model == True``, checkpoints are logged at the end of training, except when
              :paramref:`~lightning.pytorch.callbacks.Checkpoint.save_top_k` ``== -1``
              which also logs every checkpoint during training.
            * if ``log_model == False`` (default), no checkpoint is logged.

        prefix: A string to put at the beginning of metric keys.
        artifact_location: The location to store run artifacts. If not provided, the server picks an appropriate
            default.
        run_id: The run identifier of the experiment. If not provided, a new run is started.
        synchronous: Hints mlflow whether to block the execution for every logging call until complete where
            applicable. Requires mlflow >= 2.8.0

    Raises:
        ModuleNotFoundError:
            If required MLFlow package is not installed on the device.

    """

    LOGGER_JOIN_CHAR = "-"

    def __init__(
        self,
        experiment_name: str = "lightning_logs",
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = os.getenv("MLFLOW_TRACKING_URI"),
        tags: Optional[dict[str, Any]] = None,
        save_dir: Optional[str] = "./mlruns",
        log_model: Literal[True, False, "all"] = False,
        prefix: str = "",
        artifact_location: Optional[str] = None,
        run_id: Optional[str] = None,
        synchronous: Optional[bool] = None,
    ):
        _raise_enterprise_not_available()
        from pytorch_lightning_enterprise.loggers.mlflow import MLFlowLogger as EnterpriseMLFlowLogger

        super().__init__()
        self.logger_impl = EnterpriseMLFlowLogger(
            experiment_name=experiment_name,
            run_name=run_name,
            tracking_uri=tracking_uri,
            tags=tags,
            save_dir=save_dir,
            log_model=log_model,
            prefix=prefix,
            artifact_location=artifact_location,
            run_id=run_id,
            synchronous=synchronous,
        )

    @property
    @rank_zero_experiment
    def experiment(self) -> "MlflowClient":
        r"""Actual MLflow object. To use MLflow features in your :class:`~lightning.pytorch.core.LightningModule` do the
        following.

        Example::

            self.logger.experiment.some_mlflow_function()

        """
        return self.logger_impl.experiment

    @property
    def run_id(self) -> Optional[str]:
        """Create the experiment if it does not exist to get the run id.

        Returns:
            The run id.

        """
        return self.logger_impl.run_id

    @property
    def experiment_id(self) -> Optional[str]:
        """Create the experiment if it does not exist to get the experiment id.

        Returns:
            The experiment id.

        """
        return self.logger_impl.experiment_id

    @override
    @rank_zero_only
    def log_hyperparams(self, params: Union[dict[str, Any], Namespace]) -> None:
        return self.logger_impl.log_hyperparams(params)

    @override
    @rank_zero_only
    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        return self.logger_impl.log_metrics(metrics, step)

    @override
    @rank_zero_only
    def finalize(self, status: str = "success") -> None:
        return self.logger_impl.finalize(status)

    @property
    @override
    def save_dir(self) -> Optional[str]:
        """The root file directory in which MLflow experiments are saved.

        Return:
            Local path to the root experiment directory if the tracking uri is local.
            Otherwise returns `None`.

        """
        return self.logger_impl.save_dir

    @property
    @override
    def name(self) -> Optional[str]:
        """Get the experiment id.

        Returns:
            The experiment id.

        """
        return self.logger_impl.name

    @property
    @override
    def version(self) -> Optional[str]:
        """Get the run id.

        Returns:
            The run id.

        """
        return self.logger_impl.version

    @override
    def after_save_checkpoint(self, checkpoint_callback: ModelCheckpoint) -> None:
        return self.logger_impl.after_save_checkpoint(checkpoint_callback)
