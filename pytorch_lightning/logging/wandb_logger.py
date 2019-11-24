from logging import getLogger
from time import time
import os

try:
    import wandb
except ImportError:
    raise ImportError('Missing wandb package.')

from .base import LightningLoggerBase, rank_zero_only

logger = getLogger(__name__)


class WandbLogger(LightningLoggerBase):
    def __init__(self, name=None, offline=False, id=None, anonymous=False, save_dir=None,
                 version=None, project=None, tags=None, sync_checkpoints=False):
        super().__init__()
        self._name = name
        self._save_dir = save_dir or os.getcwd()
        self._anonymous = "allow" if anonymous else None
        self._id = version or id
        self._tags = tags
        self._project = project
        self._experiment = None
        self._offline = offline
        self._sync_checkpoints = sync_checkpoints

    @property
    def experiment(self):
        if self._experiment is None:
            if self._offline:
                os.environ["WANDB_MODE"] = "dryrun"
            self._experiment = wandb.init(
                name=self._name, project=self._project, anonymous=self._anonymous,
                id=self._id, resume="allow", tags=self._tags)
        return self._experiment

    def watch(self, model, log="gradients", log_freq=100):
        wandb.watch(model, log, log_freq)

    @rank_zero_only
    def log_hyperparams(self, params):
        self.experiment.config.update(params)

    @rank_zero_only
    def log_metrics(self, metrics, step_num=None):
        metrics["global_step"] = step_num
        self.experiment.history.add(metrics)

    def save(self):
        pass

    @rank_zero_only
    def finalize(self, status='success'):
        try:
            exit_code = 0 if status == 'success' else 1
            wandb.join(exit_code)
        except TypeError:
            wandb.join()

    @property
    def name(self):
        return self.experiment.project_name()

    @property
    def version(self):
        return self.experiment.id
