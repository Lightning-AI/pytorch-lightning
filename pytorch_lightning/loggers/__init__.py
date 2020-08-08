from os import environ

from pytorch_lightning.loggers.base import LightningLoggerBase, LoggerCollection
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

__all__ = [
    'LightningLoggerBase',
    'LoggerCollection',
    'TensorBoardLogger',
    'CSVLogger',
]

try:
    # needed to prevent ImportError and duplicated logs.
    environ["COMET_DISABLE_AUTO_LOGGING"] = "1"

    from pytorch_lightning.loggers.comet import CometLogger
except ImportError:  # pragma: no-cover
    del environ["COMET_DISABLE_AUTO_LOGGING"]  # pragma: no-cover
else:
    __all__.append('CometLogger')

try:
    from pytorch_lightning.loggers.mlflow import MLFlowLogger
except ImportError:  # pragma: no-cover
    pass  # pragma: no-cover
else:
    __all__.append('MLFlowLogger')

try:
    from pytorch_lightning.loggers.neptune import NeptuneLogger
except ImportError:  # pragma: no-cover
    pass  # pragma: no-cover
else:
    __all__.append('NeptuneLogger')

try:
    from pytorch_lightning.loggers.test_tube import TestTubeLogger
except ImportError:  # pragma: no-cover
    pass  # pragma: no-cover
else:
    __all__.append('TestTubeLogger')

try:
    from pytorch_lightning.loggers.wandb import WandbLogger
except ImportError:  # pragma: no-cover
    pass  # pragma: no-cover
else:
    __all__.append('WandbLogger')
