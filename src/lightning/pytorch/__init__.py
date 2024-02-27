"""Root package info."""

import logging
import os

from lightning_utilities import module_available

if os.path.isfile(os.path.join(os.path.dirname(__file__), "__about__.py")):
    from lightning.pytorch.__about__ import *  # noqa: F403
if "__version__" not in locals():
    if os.path.isfile(os.path.join(os.path.dirname(__file__), "__version__.py")):
        from lightning.pytorch.__version__ import version as __version__
    elif module_available("lightning"):
        from lightning import __version__  # noqa: F401

_root_logger = logging.getLogger()
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

# if root logger has handlers, propagate messages up and let root logger process them
if not _root_logger.hasHandlers():
    _logger.addHandler(logging.StreamHandler())
    _logger.propagate = False

from lightning.fabric.utilities.seed import seed_everything  # noqa: E402
from lightning.fabric.utilities.warnings import disable_possible_user_warnings  # noqa: E402
from lightning.pytorch.callbacks import Callback  # noqa: E402
from lightning.pytorch.core import LightningDataModule, LightningModule  # noqa: E402
from lightning.pytorch.trainer import Trainer  # noqa: E402

# this import needs to go last as it will patch other modules
import lightning.pytorch._graveyard  # noqa: E402, F401  # isort: skip

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


if os.environ.get("POSSIBLE_USER_WARNINGS", "").lower() in ("0", "off"):
    disable_possible_user_warnings()
