"""Root package info."""

import logging
from typing import Any

from pytorch_lightning import __about__
from pytorch_lightning.__about__ import *  # noqa: F401, F403

if not hasattr(__about__, "__version__"):
    from pytorch_lightning.__version__ import version as __version__  # noqa: F401

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

# if root logger has handlers, propagate messages up and let root logger process them
if not _root_logger.hasHandlers():
    _logger.addHandler(logging.StreamHandler())
    _logger.propagate = False

from lightning_fabric.utilities.seed import seed_everything  # noqa: E402
from pytorch_lightning.callbacks import Callback  # noqa: E402
from pytorch_lightning.core import LightningDataModule, LightningModule  # noqa: E402
from pytorch_lightning.trainer import Trainer  # noqa: E402

# this import needs to go last as it will patch other modules
import pytorch_lightning._graveyard  # noqa: E402, F401  # isort: skip

__all__ = ["Trainer", "LightningDataModule", "LightningModule", "Callback", "seed_everything"]

# for compatibility with namespace packages
__import__("pkg_resources").declare_namespace(__name__)

LIGHTNING_LOGO: str = """
                    ####
                ###########
             ####################
         ############################
    #####################################
##############################################
#########################  ###################
#######################    ###################
####################      ####################
##################       #####################
################        ######################
#####################        #################
######################     ###################
#####################    #####################
####################   #######################
###################  #########################
##############################################
    #####################################
         ############################
             ####################
                  ##########
                     ####
"""


def cli_lightning_logo() -> None:
    print()
    print("\033[0;35m" + LIGHTNING_LOGO + "\033[0m")
    print()
