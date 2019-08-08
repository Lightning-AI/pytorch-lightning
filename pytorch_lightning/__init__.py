try:
    from .models.trainer import Trainer
    from .root_module.root_module import LightningModule
    from .root_module.decorators import data_loader
except ImportError as e:
    print('Failing to load some internal modules.')

__version__ = '0.3.6.9'
__author__ = "William Falcon",
__author_email__ = "waf2107@columbia.edu"
__license__ = 'Apache-2'
__homepage__ = 'https://github.com/williamFalcon/pytorch-lightning',
__copyright__ = 'Copyright (c) 2018-2019, %s.' % __author__
__doc__ = """
The Keras for ML researchers using PyTorch
"""

__all__ = [
    'Trainer',
    'LightningModule',
    'data_loader',
]
