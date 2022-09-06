"""Root package info."""
import logging

from lightning_lite.__about__ import *  # noqa: F401, F403
from lightning_lite.__version__ import version as __version__  # noqa: F401


_root_logger = logging.getLogger()
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

# TODO(lite): Lite and PL interfere through this root_logger, breaking lite tests who assert through caplog
# if root logger has handlers, propagate messages up and let root logger process them
# if not _root_logger.hasHandlers():
#     _logger.addHandler(logging.StreamHandler())
#     _logger.propagate = False

from lightning_lite.lite import LightningLite  # noqa: E402
from lightning_lite.utilities.seed import seed_everything  # noqa: E402

__all__ = ["LightningLite", "seed_everything"]

# for compatibility with namespace packages
__import__("pkg_resources").declare_namespace(__name__)
