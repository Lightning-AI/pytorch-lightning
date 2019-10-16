from time import time
from logging import getLogger
from comet_ml import Experiment as CometExperiment

from .base import LightningLoggerBase, rank_zero_only


class CometLogger(LightningLoggerBase):
    def __init__(self, *args, **kwargs):
        super(CometLogger, self).__init__()
        self.comet_exp = CometExperiment(*args, **kwargs)

    @rank_zero_only
    def log_hyperparams(self, params):
        self.comet_exp.log_parameters(vars(params))

    @rank_zero_only
    def log_metrics(self, metrics, step_num):
        # self.comet_exp.set_epoch(self, metrics.get('epoch', 0))
        self.comet_exp.log_metrics(metrics)

    @rank_zero_only
    def finalize(self, status):
        self.comet_exp.end()
