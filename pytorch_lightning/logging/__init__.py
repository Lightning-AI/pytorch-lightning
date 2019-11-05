from os import environ
from .base import LightningLoggerBase, rank_zero_only

try:
    from .test_tube_logger import TestTubeLogger
except ImportError:
    pass
try:
    from .mlflow_logger import MLFlowLogger
except ImportError:
    pass
try:
    # needed to prevent ImportError and duplicated logs.
    environ["COMET_DISABLE_AUTO_LOGGING"] = "1"

    from .comet_logger import CometLogger
except ImportError:
    del environ["COMET_DISABLE_AUTO_LOGGING"]
