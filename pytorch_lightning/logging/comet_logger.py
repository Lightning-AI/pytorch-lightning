from os import environ

from comet_ml import Experiment as CometExperiment

from .base import LightningLoggerBase, rank_zero_only

# needed to prevent ImportError and duplicated logs.
environ["COMET_DISABLE_AUTO_LOGGING"] = "1"


class CometLogger(LightningLoggerBase):
    def __init__(self, *args, **kwargs):
        super(CometLogger, self).__init__()
        self.experiment = CometExperiment(*args, **kwargs)

    @rank_zero_only
    def log_hyperparams(self, params):
        self.experiment.log_parameters(vars(params))

    @rank_zero_only
    def log_metrics(self, metrics, step_num):
        # self.experiment.set_epoch(self, metrics.get('epoch', 0))
        self.experiment.log_metrics(metrics)

    @rank_zero_only
    def finalize(self, status):
        self.experiment.end()
