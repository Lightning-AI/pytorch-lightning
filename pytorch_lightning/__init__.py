"""Root package info."""

__version__ = '0.9.0'
__author__ = 'William Falcon et al.'
__author_email__ = 'waf2107@columbia.edu'
__license__ = 'Apache-2.0'
__copyright__ = 'Copyright (c) 2018-2020, %s.' % __author__
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

import logging as python_logging

_logger = python_logging.getLogger("lightning")
_logger.addHandler(python_logging.StreamHandler())
_logger.setLevel(python_logging.INFO)

try:
    # This variable is injected in the __builtins__ by the build
    # process. It used to enable importing subpackages of skimage when
    # the binaries are not built
    __LIGHTNING_SETUP__
except NameError:
    __LIGHTNING_SETUP__ = False

if __LIGHTNING_SETUP__:
    import sys  # pragma: no-cover

    sys.stdout.write(f'Partial import of `{__name__}` during the build process.\n')  # pragma: no-cover
    # We are not importing the rest of the lightning during the build process, as it may not be compiled yet
else:
    from pytorch_lightning.core import LightningDataModule, LightningModule
    from pytorch_lightning.core.step_result import TrainResult, EvalResult
    from pytorch_lightning.callbacks import Callback
    from pytorch_lightning.trainer import Trainer
    from pytorch_lightning.utilities.seed import seed_everything
    from pytorch_lightning import metrics

    __all__ = [
        'Trainer',
        'LightningDataModule',
        'LightningModule',
        'Callback',
        'seed_everything',
        'metrics',
        'EvalResult',
        'TrainResult',
    ]

    # necessary for regular bolts imports. Skip exception since bolts is not always installed
    try:
        from pytorch_lightning import bolts
    except ImportError:
        pass
    # __call__ = __all__

# for compatibility with namespace packages
__import__('pkg_resources').declare_namespace(__name__)
