from logging import getLogger
from time import time

try:
    import sacred
    from sacred.observers import MongoObserver
except ImportError:
    raise ImportError('Missing sacred package.  Run `pip install sacred`')

from pytorch_lightning.logging.base import LightningLoggerBase, rank_zero_only

logger = getLogger(__name__)


# TODO: add docstring with type definition
class SacredLogger(LightningLoggerBase):
    def __init__(self, sacred_experiment, observers):
        super().__init__()
        self.sacred_experiment = sacred_experiment
        self.experiment_name = sacred_experiment.path
        for obs in observers:
            self.sacred_experiment.observers.append(obs)

    @property
    def experiment(self):
        return self.sacred_experiment

    @property
    def run_id(self):
        try:
            return self.sacred_experiment.current_run._id
        except:
            pass
        return None

    @rank_zero_only
    def log_hyperparams(self, params):
        for k, v in vars(params).items():
            self.experiment.log_param(self.run_id, k, v)

    @rank_zero_only
    def log_metrics(self, metrics, step_num=None):
        for k, v in metrics.items():
            if isinstance(v, str):
                logger.warning(
                    f"Discarding metric with string value {k}={v}"
                )
                continue
            self.experiment.log_scalar(self.run_id, k, v, step_num)

    def save(self):
        pass

    @rank_zero_only
    def finalize(self, status="FINISHED"):
        if status == 'success':
            status = 'FINISHED'
        self.experiment.set_terminated(self.run_id, status)

    @property
    def name(self):
        return self.experiment_name

    @property
    def version(self):
        return self._run_id
