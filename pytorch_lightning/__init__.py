"""Root package info."""

import logging
import os
import sys
import time

_this_year = time.strftime("%Y")
__version__ = '1.1.7'
__author__ = 'William Falcon et al.'
__author_email__ = 'waf2107@columbia.edu'
__license__ = 'Apache-2.0'
__copyright__ = f'Copyright (c) 2018-{_this_year}, {__author__}.'
__homepage__ = 'https://github.com/PyTorchLightning/pytorch-lightning'
# this has to be simple string, see: https://github.com/pypa/twine/issues/522
__docs__ = (
    "PyTorch Lightning is the lightweight PyTorch wrapper for ML researchers."
    " Scale your models. Write less boilerplate."
)
__long_docs__ = """
Lightning is a way to organize your PyTorch code to decouple the science code from the engineering.
 It's more of a style-guide than a framework.

In Lightning, you organize your code into 3 distinct categories:

1. Research code (goes in the LightningModule).
2. Engineering code (you delete, and is handled by the Trainer).
3. Non-essential research code (logging, etc. this goes in Callbacks).

Although your research/production project might start simple, once you add things like GPU AND TPU training,
 16-bit precision, etc, you end up spending more time engineering than researching.
 Lightning automates AND rigorously tests those parts for you.

Overall, Lightning guarantees rigorously tested, correct, modern best practices for the automated parts.

Documentation
-------------
- https://pytorch-lightning.readthedocs.io/en/latest
- https://pytorch-lightning.readthedocs.io/en/stable
"""
_root_logger = logging.getLogger()
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

# if root logger has handlers, propagate messages up and let root logger process them
if not _root_logger.hasHandlers():
    _logger.addHandler(logging.StreamHandler())
    _logger.propagate = False


PACKAGE_ROOT = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)

try:
    # This variable is injected in the __builtins__ by the build
    # process. It used to enable importing subpackages of skimage when
    # the binaries are not built
    _ = None if __LIGHTNING_SETUP__ else None
except NameError:
    __LIGHTNING_SETUP__: bool = False

if __LIGHTNING_SETUP__:  # pragma: no-cover
    sys.stdout.write(f'Partial import of `{__name__}` during the build process.\n')  # pragma: no-cover
    # We are not importing the rest of the lightning during the build process, as it may not be compiled yet
else:
    from pytorch_lightning import metrics
    from pytorch_lightning.callbacks import Callback
    from pytorch_lightning.core import LightningDataModule, LightningModule
    from pytorch_lightning.trainer import Trainer
    from pytorch_lightning.utilities.seed import seed_everything

    __all__ = [
        'Trainer',
        'LightningDataModule',
        'LightningModule',
        'Callback',
        'seed_everything',
        'metrics',
    ]

# for compatibility with namespace packages
__import__('pkg_resources').declare_namespace(__name__)
