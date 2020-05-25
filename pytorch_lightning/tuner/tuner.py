from typing import Union, Optional, List, Dict, Tuple, Iterable, Any

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.tuner.training_tricks import TunerLRFinderMixin
from pytorch_lightning.tuner.lr_finder import TunerBatchScalerMixin

class Tuner(
        TunerLRFinderMixin,
        TunerBatchScalerMixin):
    
    def __init__(self, trainer: Trainer,
                 auto_scale_batch_size: Union[str, bool] = False,
                 auto_lr_find: Union[bool, str] = False):
        
        # User instance of trainer
        self.trainer = trainer
        
        # Parameters to optimize
        self.auto_scale_batch_size = auto_scale_batch_size
        self.auto_lr_find = auto_lr_find
    
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