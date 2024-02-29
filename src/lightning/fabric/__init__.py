"""Root package info."""

import logging
import os

from lightning_utilities.core.imports import package_available

if os.path.isfile(os.path.join(os.path.dirname(__file__), "__about__.py")):
    from lightning.fabric.__about__ import *  # noqa: F403
if os.path.isfile(os.path.join(os.path.dirname(__file__), "__version__.py")):
    from lightning.fabric.__version__ import version as __version__
elif package_available("lightning"):
    from lightning import __version__  # noqa: F401

_root_logger = logging.getLogger()
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

if not _root_logger.hasHandlers():
    _logger.addHandler(logging.StreamHandler())
    _logger.propagate = False


# In PyTorch 2.0+, setting this variable will force `torch.cuda.is_available()` and `torch.cuda.device_count()`
# to use an NVML-based implementation that doesn't poison forks.
# https://github.com/pytorch/pytorch/issues/83973
os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"


from lightning.fabric.fabric import Fabric  # noqa: E402
from lightning.fabric.utilities.seed import seed_everything  # noqa: E402
from lightning.fabric.utilities.warnings import disable_possible_user_warnings  # noqa: E402
from lightning.fabric.wrappers import is_wrapped  # noqa: E402

# this import needs to go last as it will patch other modules
import lightning.fabric._graveyard  # noqa: E402, F401  # isort: skip

__all__ = ["Fabric", "seed_everything", "is_wrapped"]

# for compatibility with namespace packages
__import__("pkg_resources").declare_namespace(__name__)


if os.environ.get("POSSIBLE_USER_WARNINGS", "").lower() in ("0", "off"):
    disable_possible_user_warnings()
