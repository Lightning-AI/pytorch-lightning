"""Root package info."""

import logging
import os

from lightning_utilities.core.imports import module_available, package_available

_root_logger = logging.getLogger()
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

_console = logging.StreamHandler()
_console.setLevel(logging.INFO)

formatter = logging.Formatter("%(levelname)s: %(message)s")
_console.setFormatter(formatter)

# if root logger has handlers, propagate messages up and let root logger process them,
# otherwise use our own handler
if not _root_logger.hasHandlers():
    _logger.addHandler(_console)
    _logger.propagate = False


if os.path.isfile(os.path.join(os.path.dirname(__file__), "__about__.py")):
    from lightning.app.__about__ import *  # noqa: F403
if "__version__" not in locals():
    if os.path.isfile(os.path.join(os.path.dirname(__file__), "__version__.py")):
        from lightning.app.__version__ import version as __version__
    elif package_available("lightning"):
        from lightning import __version__  # noqa: F401

from lightning.app.core.app import LightningApp  # noqa: E402
from lightning.app.core.flow import LightningFlow  # noqa: E402
from lightning.app.core.work import LightningWork  # noqa: E402
from lightning.app.plugin.plugin import LightningPlugin  # noqa: E402
from lightning.app.utilities.packaging.build_config import BuildConfig  # noqa: E402
from lightning.app.utilities.packaging.cloud_compute import CloudCompute  # noqa: E402

if module_available("lightning.app.components.demo"):
    from lightning.app.components import demo  # noqa: F401

__package_name__ = "lightning.app".split(".")[0]

_PACKAGE_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_PACKAGE_ROOT))
if __package_name__ == "lightning":
    _PACKAGE_ROOT = os.path.dirname(_PACKAGE_ROOT)
    _PROJECT_ROOT = os.path.dirname(_PROJECT_ROOT)

__all__ = ["LightningApp", "LightningFlow", "LightningWork", "LightningPlugin", "BuildConfig", "CloudCompute"]
