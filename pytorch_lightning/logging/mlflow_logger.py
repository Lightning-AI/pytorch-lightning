from time import time
from logging import getLogger

import mlflow

from .base import LightningLoggerBase, rank_zero_only

logger = getLogger(__name__)


class MLFlowLogger(LightningLoggerBase):

    def __init__(self, experiment_name, tracking_uri=None):
        super().__init__()
        self.client = mlflow.tracking.MlflowClient(tracking_uri)

        experiment = self.client.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.warning(
                f"Experiment with name f{experiment_name} not found. Creating it."
            )
            self.client.create_experiment(experiment_name)
            experiment = self.client.get_experiment_by_name(experiment_name)

        run = self.client.create_run(experiment.experiment_id)
        self.run_id = run.info.run_id

    @rank_zero_only
    def log_hyperparams(self, params):
        for k, v in vars(params).items():
            self.client.log_param(self.run_id, k, v)

    @rank_zero_only
    def log_metrics(self, metrics, step_num=None):
        timestamp_ms = int(time() * 1000)
        for k, v in metrics.items():
            self.client.log_metric(self.run_id, k, v, timestamp_ms, step_num)

    def save(self):
        pass

    @rank_zero_only
    def finalize(self, status="FINISHED"):
        self.client.set_terminated(self.run_id, status)
