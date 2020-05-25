from typing import Union, Optional, List, Dict, Tuple, Iterable, Any

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.tuner.training_tricks import TunerLRFinderMixin
from pytorch_lightning.tuner.lr_finder import TunerBatchScalerMixin

class Tuner(
        TunerLRFinderMixin,
        TunerBatchScalerMixin):
    r"""
    Flags for enabling and disabling automatic tuner algorithms. For manual control
    call each tuner algorithm by its own method.
    
    Args:
        auto_lr_find: If set to True, will `initially` run a learning rate finder,
            trying to optimize initial learning for faster convergence. Sets learning
            rate in self.lr or self.learning_rate in the LightningModule.
            To use a different key, set a string instead of True with the key name.

        auto_scale_batch_size: If set to True, will `initially` run a batch size
            finder trying to find the largest batch size that fits into memory.
            The result will be stored in self.batch_size in the LightningModule.
            Additionally, can be set to either `power` that estimates the batch size through
            a power search or `binsearch` that estimates the batch size through a binary search.
    """
    
    def __init__(self, trainer: Trainer,
                 auto_scale_batch_size: Union[str, bool] = False,
                 auto_lr_find: Union[bool, str] = False):
        
        # User instance of trainer
        self.trainer = trainer
        
        # Parameters to optimize
        self.auto_scale_batch_size = auto_scale_batch_size
        self.auto_lr_find = auto_lr_find
        
        # For checking dependency
        self._scale_batch_size_called = False
        self._lr_find_called = False
    
    def optimize(self, 
                 model: LightningModule,
                 train_dataloader = train_dataloader
                 ):    
        # Run auto batch size scaling
        if self.auto_scale_batch_size:      
            self.scale_batch_size(model)
            
        # Run learning rate finder:
        if self.auto_lr_find:
            self.lr_find(model)