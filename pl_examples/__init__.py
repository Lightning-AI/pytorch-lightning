import os

from six.moves import urllib

from pytorch_lightning.utilities import _module_available

# TorchVision hotfix https://github.com/pytorch/vision/issues/1938
opener = urllib.request.build_opener()
opener.addheaders = [("User-agent", "Mozilla/5.0")]
urllib.request.install_opener(opener)

_EXAMPLES_ROOT = os.path.dirname(__file__)
_PACKAGE_ROOT = os.path.dirname(_EXAMPLES_ROOT)
_DATASETS_PATH = os.path.join(_PACKAGE_ROOT, "Datasets")

_DALI_AVAILABLE = _module_available("nvidia.dali")

LIGHTNING_LOGO = """
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


def nice_print(msg, last=False):
    print()
    print("\033[0;35m" + msg + "\033[0m")
    if last:
        print()


def cli_lightning_logo():
    nice_print(LIGHTNING_LOGO)
