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
    """
    Logger for W&B.

    Args:
        name (str): display name for the run.
        save_dir (str): path where data is saved.
        offline (bool): run offline (data can be streamed later to wandb servers).
        id or version (str): sets the version, mainly used to resume a previous run.
        anonymous (bool): enables or explicitly disables anonymous logging.
        project (str): the name of the project to which this run will belong.
        tags (list of str): tags associated with this run.
    """

    def __init__(self, name=None, save_dir=None, offline=False, id=None, anonymous=False,
                 version=None, project=None, tags=None):
        super().__init__()
        self._name = name
        self._save_dir = save_dir
        self._anonymous = "allow" if anonymous else None
        self._id = version or id
        self._tags = tags
        self._project = project
        self._experiment = None
        self._offline = offline

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
