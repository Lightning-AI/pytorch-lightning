"""Root package info."""
import logging
import os

from lightning_utilities.core.imports import module_available

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


from lightning_app import __about__  # noqa: E402
from lightning_app import components  # noqa: E402, F401
from lightning_app.__about__ import *  # noqa: E402, F401, F403

if not hasattr(__about__, "__version__"):
    from lightning_app.__version__ import version as __version__  # noqa: F401

from lightning_app.core.app import LightningApp  # noqa: E402
from lightning_app.core.flow import LightningFlow  # noqa: E402
from lightning_app.core.work import LightningWork  # noqa: E402
from lightning_app.perf import pdb  # noqa: E402
from lightning_app.plugin.plugin import LightningPlugin  # noqa: E402
from lightning_app.utilities.packaging.build_config import BuildConfig  # noqa: E402
from lightning_app.utilities.packaging.cloud_compute import CloudCompute  # noqa: E402

if module_available("lightning_app.components.demo"):
    from lightning_app.components import demo  # noqa: F401

__package_name__ = "lightning_app".split(".")[0]

_PACKAGE_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_PACKAGE_ROOT))

__all__ = ["LightningApp", "LightningFlow", "LightningWork", "LightningPlugin", "BuildConfig", "CloudCompute", "pdb"]
