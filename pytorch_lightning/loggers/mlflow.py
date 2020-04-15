"""
Log using `mlflow <https://mlflow.org>`_

.. code-block:: python

    from pytorch_lightning.loggers import MLFlowLogger
    mlf_logger = MLFlowLogger(
        experiment_name="default",
        tracking_uri="file:/."
    )
    trainer = Trainer(logger=mlf_logger)


Use the logger anywhere in you LightningModule as follows:

.. code-block:: python

    def train_step(...):
        # example
        self.logger.experiment.whatever_ml_flow_supports(...)

    def any_lightning_module_function_or_hook(...):
        self.logger.experiment.whatever_ml_flow_supports(...)

"""
import os
from argparse import Namespace
from time import time
from typing import Optional, Dict, Any, Union

try:
    import mlflow
    from mlflow.tracking import MlflowClient
except ImportError:  # pragma: no-cover
    raise ImportError('You want to use `mlflow` logger which is not installed yet,'  # pragma: no-cover
                      ' install it with `pip install mlflow`.')

from pytorch_lightning import _logger as log
from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_only


class MLFlowLogger(LightningLoggerBase):
    """MLFLow logger"""

    def __init__(self,
                 experiment_name: str = 'default',
                 tracking_uri: Optional[str] = None,
                 tags: Optional[Dict[str, Any]] = None,
                 save_dir: Optional[str] = None):
        r"""
        Logs using MLFlow

        Args:
            experiment_name (str): The name of the experiment
            tracking_uri (str): where this should track
            tags (dict): todo this param
        """
        super().__init__()
        if not tracking_uri and save_dir:
            tracking_uri = f'file:{os.sep * 2}{save_dir}'
        self._mlflow_client = MlflowClient(tracking_uri)
        self.experiment_name = experiment_name
        self._run_id = None
        self.tags = tags

    @property
    def experiment(self) -> MlflowClient:
        r"""
        Actual mlflow object. To use mlflow features do the following.

        Example::

            self.logger.experiment.some_mlflow_function()

        """
        return self._mlflow_client

    @property
    def run_id(self):
        if self._run_id is not None:
            return self._run_id

        expt = self._mlflow_client.get_experiment_by_name(self.experiment_name)

        if expt:
            self._expt_id = expt.experiment_id
        else:
            log.warning(f'Experiment with name {self.experiment_name} not found. Creating it.')
            self._expt_id = self._mlflow_client.create_experiment(name=self.experiment_name)

        run = self._mlflow_client.create_run(experiment_id=self._expt_id, tags=self.tags)
        self._run_id = run.info.run_id
        return self._run_id

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = self._convert_params(params)
        params = self._flatten_dict(params)
        for k, v in params.items():
            self.experiment.log_param(self.run_id, k, v)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        timestamp_ms = int(time() * 1000)
        for k, v in metrics.items():
            if isinstance(v, str):
                log.warning(f'Discarding metric with string value {k}={v}.')
                continue
            self.experiment.log_metric(self.run_id, k, v, timestamp_ms, step)

    @rank_zero_only
    def finalize(self, status: str = 'FINISHED') -> None:
        super().finalize(status)
        if status == 'success':
            status = 'FINISHED'
        self.experiment.set_terminated(self.run_id, status)

    @property
    def name(self) -> str:
        return self.experiment_name

    @property
    def version(self) -> str:
        return self._run_id
