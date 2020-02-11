"""Root package info."""

__version__ = '0.6.1.dev'
__author__ = 'William Falcon et al.'
__author_email__ = 'waf2107@columbia.edu'
__license__ = 'Apache-2.0'
__copyright__ = 'Copyright (c) 2018-2020, %s.' % __author__
__homepage__ = 'https://github.com/PyTorchLightning/pytorch-lightning'
# this has to be simple string, see: https://github.com/pypa/twine/issues/522
__docs__ = "PyTorch Lightning is the lightweight PyTorch wrapper for ML researchers." \
           " Scale your models. Write less boilerplate."

try:
    # This variable is injected in the __builtins__ by the build
    # process. It used to enable importing subpackages of skimage when
    # the binaries are not built
    __LIGHTNING_SETUP__
except NameError:
    __LIGHTNING_SETUP__ = False

if __LIGHTNING_SETUP__:
    import sys
    sys.stderr.write('Partial import of `torchlightning` during the build process.\n')
    # We are not importing the rest of the scikit during the build
    # process, as it may not be compiled yet
else:
    import logging as log
    log.basicConfig(level=log.INFO)

    from .core import data_loader, LightningModule
    from .trainer import Trainer

    __all__ = [
        'Trainer',
        'LightningModule',
        'data_loader',
    ]
    # __call__ = __all__
