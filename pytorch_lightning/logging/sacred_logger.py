from logging import getLogger
from time import time

try:
    import sacred
except ImportError:
    raise ImportError('Missing sacred package.')

from .base import LightningLoggerBase, rank_zero_only

logger = getLogger(__name__)


class SacredLogger(LightningLoggerBase):
    def __init__(self, database_location, database_name, experiment_name):
        super().__init__()
        self.experiment = sacred.Experiment(experiment_name)
        self.experiment_name = experiment_name
        self._run_id = None

    @property
    def run_id(self):
        if self._run_id is not None:
            return self._run_id

        experiment = self.experiment.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            logger.warning(
                f"Experiment with name f{self.experiment_name} not found. Creating it."
            )
            self.experiment.create_experiment(self.experiment_name)
            experiment = self.experiment.get_experiment_by_name(self.experiment_name)

        run = self.experiment.create_run(experiment.experiment_id, tags=self.tags)
        self._run_id = run.info.run_id
        return self._run_id

    @rank_zero_only
    def log_hyperparams(self, params):
        for k, v in vars(params).items():
            self.experiment.log_param(self.run_id, k, v)

    @rank_zero_only
    def log_metrics(self, metrics, step_num=None):
        timestamp_ms = int(time() * 1000)
        for k, v in metrics.items():
            if isinstance(v, str):
                logger.warning(
                    f"Discarding metric with string value {k}={v}"
                )
                continue
            self.experiment.log_metric(self.run_id, k, v, timestamp_ms, step_num)

    def save(self):
        pass

    @rank_zero_only
    def finalize(self, status="FINISHED"):
        if status == 'success':
            status = 'FINISHED'
        self.experiment.set_terminated(self.run_id, status)
