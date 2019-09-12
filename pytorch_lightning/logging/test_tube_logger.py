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
        self.close()

    def close(self):
        self.experiment.close()

    @property
    def rank(self):
        return self.experiment.rank

    @rank.setter
    def rank(self, value):
        self.experiment.rank = value

    @property
    def version(self):
        return self.experiment.version

    # Test tube experiments are not pickleable, so we need to override a few
    # methods to get DDP working. See
    # https://docs.python.org/3/library/pickle.html#handling-stateful-objects
    # for more info.
    def __getstate__(self):
        state = self.__dict__.copy()
        state["experiment"] = self.experiment.get_meta_copy()
        return state

    def __setstate__(self, state):
        self.experiment = state["experiment"].get_non_ddp_exp()
        del state['experiment']
        self.__dict__.update(state)
