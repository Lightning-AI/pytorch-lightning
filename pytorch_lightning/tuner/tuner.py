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
        self.auto_lr_find = auto_lr_find
        self.auto_scale_batch_size = auto_scale_batch_size
    
    def optimize(self, 
                 model: LightningModule,
                 ):
        pass
    
        # Run auto batch size scaling
        if self.auto_scale_batch_size:
            if isinstance(self.auto_scale_batch_size, bool):
                self.auto_scale_batch_size = 'power'
            self.scale_batch_size(model, mode=self.auto_scale_batch_size)
            model.logger = self.logger  # reset logger binding

        # Run learning rate finder:
        if self.auto_lr_find:
            self._run_lr_finder_internally(model)
            model.logger = self.logger  # reset logger binding