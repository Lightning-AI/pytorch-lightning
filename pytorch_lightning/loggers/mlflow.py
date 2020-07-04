"""
MLflow
------
"""
from argparse import Namespace
from time import time
from typing import Optional, Dict, Any, Union

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    _MLFLOW_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no-cover
    mlflow = None
    MlflowClient = None
    _MLFLOW_AVAILABLE = False


from pytorch_lightning import _logger as log
from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only


LOCAL_FILE_URI_PREFIX = "file:"


class MLFlowLogger(LightningLoggerBase):
    """
    Log using `MLflow <https://mlflow.org>`_. Install it with pip:

    .. code-block:: bash

        pip install mlflow

    Example:
        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.loggers import MLFlowLogger
        >>> mlf_logger = MLFlowLogger(
        ...     experiment_name="default",
        ...     tracking_uri="file:./ml-runs"
        ... )
        >>> trainer = Trainer(logger=mlf_logger)

    Use the logger anywhere in you :class:`~pytorch_lightning.core.lightning.LightningModule` as follows:

    >>> from pytorch_lightning import LightningModule
    >>> class LitModel(LightningModule):
    ...     def training_step(self, batch, batch_idx):
    ...         # example
    ...         self.logger.experiment.whatever_ml_flow_supports(...)
    ...
    ...     def any_lightning_module_function_or_hook(self):
    ...         self.logger.experiment.whatever_ml_flow_supports(...)

    Args:
        experiment_name: The name of the experiment
        tracking_uri: Address of local or remote tracking server.
            If not provided, defaults to the service set by ``mlflow.tracking.set_tracking_uri``.
        tags: A dictionary tags for the experiment.
        save_dir: A path to a local directory where the MLflow runs get saved.
            Defaults to `file:/<tracking_uri>` if <tracking_uri> is

    """

    def __init__(self,
                 experiment_name: str = 'default',
                 tracking_uri: Optional[str] = None,
                 tags: Optional[Dict[str, Any]] = None,
                 save_dir: Optional[str] = None):

        if not _MLFLOW_AVAILABLE:
            raise ImportError('You want to use `mlflow` logger which is not installed yet,'
                              ' install it with `pip install mlflow`.')
        super().__init__()
        if not tracking_uri and save_dir:
            tracking_uri = f'{LOCAL_FILE_URI_PREFIX}{save_dir}'

        self._experiment_name = experiment_name
        self.tags = tags
        self._tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self._experiment_name)
        exp = mlflow.get_experiment_by_name(self._experiment_name)
        self._experiment_id = exp.experiment_id
        run = mlflow.active_run() or mlflow.start_run(experiment_id=exp.experiment_id)
        self._run_id = run.info.run_id
        self._mlflow_client = MlflowClient(self._tracking_uri)

    @property
    def save_dir(self):
        """
        The root file directory in which MLflow experiments are saved.

        Return:
            Local path to the root experiment directory if the tracking uri is local.
            Otherwhise returns `None`.
        """
        if self._tracking_uri.startswith(LOCAL_FILE_URI_PREFIX):
            return self._tracking_uri.lstrip(LOCAL_FILE_URI_PREFIX)

    @property
    @rank_zero_experiment
    def experiment(self) -> MlflowClient:
        r"""
        Actual MLflow object. To use MLflow features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.

        Example::

            self.logger.experiment.some_mlflow_function()

        """
        return self._mlflow_client

    @property
    def run_id(self):
        return self._run_id

    @property
    def experiment_id(self):
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
            self.experiment.log_metric(self.run_id, k, v, timestamp_ms, step)

    @rank_zero_only
    def finalize(self, status: str = 'FINISHED') -> None:
        super().finalize(status)
        status = 'FINISHED' if status == 'success' else status
        if self.experiment.get_run(self.run_id):
            self.experiment.set_terminated(self.run_id, status)

    @property
    def name(self) -> str:
        return self.experiment_id

    @property
    def version(self) -> str:
        return self.run_id
