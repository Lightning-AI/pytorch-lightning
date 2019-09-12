import os.path
from copy import copy

from .base import LightningLoggerBase, rank_zero_only

from test_tube import Experiment


class TestTubeLogger(LightningLoggerBase):
    def __init__(
        self, save_dir, name="default", debug=False, version=None, create_git_tag=False
    ):
        super().__init__()
        self.experiment = Experiment(
            save_dir=save_dir,
            name=name,
            debug=debug,
            version=version,
            create_git_tag=create_git_tag,
        )

    @rank_zero_only
    def log_hyperparams(self, params):
        self.experiment.argparse(params)

    @rank_zero_only
    def log_metrics(self, metrics, step_num=None):
        self.experiment.log(metrics, global_step=step_num)

    @rank_zero_only
    def save(self):
        self.experiment.save()

    @rank_zero_only
    def finalize(self, status):
        self.save()
    
    @property
    def rank(self):
        return self.experiment.rank
    
    @rank.setter
    def rank(self, value):
        self.experiment.rank = value

    @property
    def version(self):
        return self.experiment.version

    def _convert(self, val):
        constructors = [int, float, str]

        if type(val) is str:
            if val.lower() == 'true':
                return True
            if val.lower() == 'false':
                return False

        for c in constructors:
            try:
                return c(val)
            except ValueError:
                pass
        return va
