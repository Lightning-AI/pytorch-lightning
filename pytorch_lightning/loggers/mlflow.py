# Copyright The PyTorch Lightning team.
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
import re
import warnings
from argparse import Namespace
from time import time
from typing import Any, Dict, Optional, Union

try:
    import mlflow
    from mlflow.tracking import MlflowClient
except ModuleNotFoundError:  # pragma: no-cover
    mlflow = None
    MlflowClient = None


from pytorch_lightning import _logger as log
from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only

LOCAL_FILE_URI_PREFIX = "file:"


class MLFlowLogger(LightningLoggerBase):
    """
    Log using `MLflow <https://mlflow.org>`_.

    Install it with pip:

    .. code-block:: bash

        pip install mlflow

    .. code-block:: python

        from pytorch_lightning import Trainer
        from pytorch_lightning.loggers import MLFlowLogger
        mlf_logger = MLFlowLogger(
            experiment_name="default",
            tracking_uri="file:./ml-runs"
        )
        trainer = Trainer(logger=mlf_logger)

    Use the logger anywhere in your :class:`~pytorch_lightning.core.lightning.LightningModule` as follows:

    .. code-block:: python

        from pytorch_lightning import LightningModule
        class LitModel(LightningModule):
            def training_step(self, batch, batch_idx):
                # example
                self.logger.experiment.whatever_ml_flow_supports(...)

            def any_lightning_module_function_or_hook(self):
                self.logger.experiment.whatever_ml_flow_supports(...)

    Args:
        experiment_name: The name of the experiment
        tracking_uri: Address of local or remote tracking server.
            If not provided, defaults to `file:<save_dir>`.
        tags: A dictionary tags for the experiment.
        save_dir: A path to a local directory where the MLflow runs get saved.
            Defaults to `./mlflow` if `tracking_uri` is not provided.
            Has no effect if `tracking_uri` is provided.

    """

    def __init__(
        self,
        experiment_name: str = 'default',
        tracking_uri: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        save_dir: Optional[str] = './mlruns'
    ):
        if mlflow is None:
            raise ImportError('You want to use `mlflow` logger which is not installed yet,'
                              ' install it with `pip install mlflow`.')
        super().__init__()
        if not tracking_uri:
            tracking_uri = f'{LOCAL_FILE_URI_PREFIX}{save_dir}'

        self._experiment_name = experiment_name
        self._experiment_id = None
        self._tracking_uri = tracking_uri
        self._run_id = None
        self.tags = tags
        self._mlflow_client = MlflowClient(tracking_uri)

    @property
    @rank_zero_experiment
    def experiment(self) -> MlflowClient:
        r"""
        Actual MLflow object. To use MLflow features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.

        Example::

            self.logger.experiment.some_mlflow_function()

        """
        if self._experiment_id is None:
            expt = self._mlflow_client.get_experiment_by_name(self._experiment_name)
            if expt is not None:
                self._experiment_id = expt.experiment_id
            else:
                log.warning(f'Experiment with name {self._experiment_name} not found. Creating it.')
                self._experiment_id = self._mlflow_client.create_experiment(name=self._experiment_name)

        if self._run_id is None:
            run = self._mlflow_client.create_run(experiment_id=self._experiment_id, tags=self.tags)
            self._run_id = run.info.run_id
        return self._mlflow_client

    @property
    def run_id(self):
        # create the experiment if it does not exist to get the run id
        _ = self.experiment
        return self._run_id

    @property
    def experiment_id(self):
        # create the experiment if it does not exist to get the experiment id
        _ = self.experiment
        return self._experiment_id

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = self._convert_params(params)
        params = self._flatten_dict(params)
        for k, v in params.items():
            self.experiment.log_param(self.run_id, k, v)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        assert rank_zero_only.rank == 0, 'experiment tried to log from global_rank != 0'

        timestamp_ms = int(time() * 1000)
        for k, v in metrics.items():
            if isinstance(v, str):
                log.warning(f'Discarding metric with string value {k}={v}.')
                continue

            new_k = re.sub("[^a-zA-Z0-9_/. -]+", "", k)
            if k != new_k:
                warnings.warn(("MLFlow only allows '_', '/', '.' and ' ' special characters in metric name.\n",
                               f"Replacing {k} with {new_k}."))
            k = new_k

            self.experiment.log_metric(self.run_id, k, v, timestamp_ms, step)

    @rank_zero_only
    def finalize(self, status: str = 'FINISHED') -> None:
        super().finalize(status)
        status = 'FINISHED' if status == 'success' else status
        if self.experiment.get_run(self.run_id):
            self.experiment.set_terminated(self.run_id, status)

    @property
    def save_dir(self) -> Optional[str]:
        """
        The root file directory in which MLflow experiments are saved.

        Return:
            Local path to the root experiment directory if the tracking uri is local.
            Otherwhise returns `None`.
        """
        if self._tracking_uri.startswith(LOCAL_FILE_URI_PREFIX):
            return self._tracking_uri.lstrip(LOCAL_FILE_URI_PREFIX)

    @property
    def name(self) -> str:
        return self.experiment_id

    @property
    def version(self) -> str:
        return self.run_id
