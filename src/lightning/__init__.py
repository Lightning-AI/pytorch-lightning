"""Root package info."""
import logging

from lightning_utilities import module_available

# explicitly don't set root logger's propagation and leave this to subpackages to manage
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

_console = logging.StreamHandler()
_console.setLevel(logging.INFO)

formatter = logging.Formatter("%(levelname)s: %(message)s")
_console.setFormatter(formatter)
_logger.addHandler(_console)

from lightning.__about__ import *  # noqa: E402, F401, F403
from lightning.__version__ import version as __version__  # noqa: E402, F401
from lightning.fabric.fabric import Fabric  # noqa: E402
from lightning.fabric.utilities.seed import seed_everything  # noqa: E402
from lightning.pytorch.callbacks import Callback  # noqa: E402
from lightning.pytorch.core import LightningDataModule, LightningModule  # noqa: E402
from lightning.pytorch.trainer import Trainer  # noqa: E402

__all__ = [
    "Trainer",
    "LightningDataModule",
    "LightningModule",
    "Callback",
    "seed_everything",
    "Fabric",
]

# todo: try to parse if the foiling is because of dependency or just wrong import
_ABLE_LOAD_APP = module_available("lightning.app")
_ABLE_LOAD_DATA = module_available("lightning.data")


if _ABLE_LOAD_APP:
    from lightning.app import storage
    from lightning.app.core.app import LightningApp
    from lightning.app.core.flow import LightningFlow
    from lightning.app.core.work import LightningWork
    from lightning.app.utilities.packaging.build_config import BuildConfig
    from lightning.app.utilities.packaging.cloud_compute import CloudCompute

    __all__ += ["LightningApp", "LightningFlow", "LightningWork", "BuildConfig", "CloudCompute", "storage"]


if _ABLE_LOAD_DATA:
    from lightning.data import LightningDataset, LightningIterableDataset

    __all__ += ["LightningDataset", "LightningIterableDataset"]
