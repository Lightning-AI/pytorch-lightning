"""Root package info."""
import logging
from typing import Any

_DETAIL = 15  # between logging.INFO and logging.DEBUG, used for logging in production use cases


def _detail(self: Any, message: str, *args: Any, **kwargs: Any) -> None:
    if self.isEnabledFor(_DETAIL):
        # logger takes its '*args' as 'args'
        self._log(_DETAIL, message, args, **kwargs)


logging.addLevelName(_DETAIL, "DETAIL")
logging.detail = _detail
logging.Logger.detail = _detail

_root_logger = logging.getLogger()
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

_console = logging.StreamHandler()
_console.setLevel(logging.INFO)

formatter = logging.Formatter("%(levelname)s: %(message)s")
_console.setFormatter(formatter)

# if root logger has handlers, propagate messages up and let root logger process them
if not _root_logger.hasHandlers():
    _logger.addHandler(_console)
    _logger.propagate = False


from lightning.__about__ import *  # noqa: E402, F401, F403
from lightning.__version__ import version as __version__  # noqa: E402, F401
from lightning.app.core.app import LightningApp  # noqa: E402
from lightning.app.core.flow import LightningFlow  # noqa: E402
from lightning.app.core.work import LightningWork  # noqa: E402
from lightning.app.utilities.packaging.build_config import BuildConfig  # noqa: E402
from lightning.app.utilities.packaging.cloud_compute import CloudCompute  # noqa: E402
from lightning.pytorch.callbacks import Callback  # noqa: E402
from lightning.pytorch.core import LightningDataModule, LightningModule  # noqa: E402
from lightning.pytorch.trainer import Trainer  # noqa: E402
from lightning.pytorch.utilities.seed import seed_everything  # noqa: E402

__all__ = [
    "LightningApp",
    "LightningFlow",
    "LightningWork",
    "BuildConfig",
    "CloudCompute",
    "Trainer",
    "LightningDataModule",
    "LightningModule",
    "Callback",
    "seed_everything",
]
