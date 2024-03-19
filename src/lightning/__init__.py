"""Root package info."""

import logging
import sys

# explicitly don't set root logger's propagation and leave this to subpackages to manage
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

_console = logging.StreamHandler()
_console.setLevel(logging.INFO)

formatter = logging.Formatter("%(levelname)s: %(message)s")
_console.setFormatter(formatter)
_logger.addHandler(_console)

from lightning.__about__ import *  # noqa: E402, F403
from lightning.__version__ import version as __version__  # noqa: E402
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
    "__version__",
]


def _cli_entry_point() -> None:
    from lightning_utilities.core.imports import ModuleAvailableCache, RequirementCache

    if not (
        ModuleAvailableCache("lightning.app")
        if RequirementCache("lightning-utilities<0.10.0")
        else RequirementCache(module="lightning.app")
    ):
        print("The `lightning` command requires additional dependencies: `pip install lightning[app]`")
        sys.exit(1)

    from lightning.app.cli.lightning_cli import main

    main()
