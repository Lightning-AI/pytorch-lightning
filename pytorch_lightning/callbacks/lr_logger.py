from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_warn


class LearningRateLogger(LearningRateMonitor):
    def __init__(self, *args, **kwargs):
        rank_zero_warn("`LearningRateLogger` is now `LearningRateMonitor`"
                       " and this will be removed in v0.11.0", DeprecationWarning)
        super().__init__(*args, **kwargs)
