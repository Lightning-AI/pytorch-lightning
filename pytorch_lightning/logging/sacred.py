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
    def __init__(self, sacred_experiment):
        super().__init__()
        self.sacred_experiment = sacred_experiment
        self.experiment_name = sacred_experiment.path
        self._run_id = None

    @property
    def experiment(self):
        return self.sacred_experiment

    @property
    def run_id(self):
        if self._run_id is not None:
            return self._run_id

        # self._run_id = self.sacred_experiment.current_run.info.run_id
        # TODO how to get run_id?
        print(self.sacred_experiment.current_run.info)
        self._run_id = 0
        return self._run_id

    @rank_zero_only
    def log_hyperparams(self, params):
        # probably not needed bc. it is dealt with by sacred
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        for k, v in metrics.items():
            if isinstance(v, str):
                logger.warning(
                    f"Discarding metric with string value {k}={v}"
                )
                continue
            self.experiment.log_scalar(k, v, step)

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
        return self.run_id
