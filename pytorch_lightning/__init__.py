from .root_module.decorators import data_loader
from .root_module.root_module import LightningModule
from .trainer.trainer import Trainer

__all__ = [
    'Trainer',
    'LightningModule',
    'data_loader',
]

__version__ = '0.5.2.1'
__author__ = ' William Falcon et al.'
__author_email__ = 'waf2107@columbia.edu'
__license__ = 'Apache-2.0'
__homepage__ = 'https://github.com/williamFalcon/pytorch-lightning',
__doc__ = """# PyTorch Lightning

The lightweight PyTorch wrapper for ML researchers. Scale your models. Write less boilerplate.
"""
