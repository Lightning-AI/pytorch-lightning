from os import environ
from .base import LightningLoggerBase, rank_zero_only

try:
    from .test_tube_logger import TestTubeLogger
except ModuleNotFoundError:
    pass
try:
    from .mlflow_logger import MLFlowLogger
except ModuleNotFoundError:
    pass
try:
    from .comet_logger import CometLogger

    # needed to prevent ImportError and duplicated logs.
    environ["COMET_DISABLE_AUTO_LOGGING"] = "1"
except ModuleNotFoundError:
    pass
