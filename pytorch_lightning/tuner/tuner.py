from typing import Union, Optional, List, Dict, Tuple, Iterable, Any

from torch.utils.data import DataLoader

from pytorch_lightning import _logger as log
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.tuner.lr_finder import TunerLRFinderMixin
from pytorch_lightning.tuner.batch_scaler import TunerBatchScalerMixin
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.warnings import ExperimentalWarning
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.parsing import nested_hasattr, nested_setattr

class Tuner(TunerLRFinderMixin, TunerBatchScalerMixin):
    r"""
    Flags for enabling and disabling automatic tuner algorithms. For manual control
    call each tuner algorithm by its own method.
    
    Args:
        auto_lr_find: If set to True, will run a learning rate finder,
            trying to optimize initial learning for faster convergence. Sets learning
            rate in self.lr or self.learning_rate in the LightningModule.
            To use a different key, set a string instead of True with the field name.

        auto_scale_batch_size: If set to True, will run a batch size scaler
            trying to find the largest batch size that fits into memory.
            The result will be stored in self.lr or self.batch_size in the LightningModule.
            To use a different key, set a string instead of True with the field name.
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
        
        rank_zero_warn('Tuner class is `experimental`, meaning that some of'
                       ' its functionality is under tested and its interface may'
                       ' change drastically within the next few releases',
                       ExperimentalWarning)
    
    def optimize(self, 
                 model: LightningModule,
                 train_dataloader: Optional[DataLoader] = None,
                 val_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None
                 ):
        r"""
        Automatic run the enabled tuner algorithms

        Args:
            model: Model to fit.

            train_dataloader: A Pytorch
                DataLoader with training samples. If the model has
                a predefined train_dataloader method this will be skipped.

            val_dataloaders: Either a single
                Pytorch Dataloader or a list of them, specifying validation samples.
                If the model has a predefined val_dataloaders method this will be skipped

        Example::

            # TODO example

        """
        # Bind train_dataloader and val_dataloader to trainer
        self.trainer.__attach_dataloaders(model, train_dataloader, val_dataloaders)
        
        # Run auto batch size scaling
        if self.auto_scale_batch_size:
            self._call_internally(model, 
                                  self.scale_batch_size, 
                                  self.auto_scale_batch_size,
                                  ['batch_size', 'bs'])
            
        # Run learning rate finder:
        if self.auto_lr_find:
            self._call_internally(model, 
                                  self.lr_find, 
                                  self.auto_lr_find, 
                                  ['learning_rate', 'lr'])
        
        # Reset model logger
        model.logger = self.trainer.logger
    
    def _call_internally(self, model, method, attribute, default):
        attribute = [attribute] if isinstance(attribute, str) else default
        
        # Check that user has the wanted attribute in their model
        found = False
        for a in attribute:
            if nested_hasattr(model, a):
                arg, found = a, True
        if not found:
            raise MisconfigurationException('Model does not have a field called'
                    f' {attribute} which is required by tuner algorithm {method}')
        
        # Call method
        obj = method(model, attribute_name=arg)
        
        # Get suggested value
        value = obj.suggestion()
        
        # Set value in model
        nested_setattr(model, arg, value)
        log.info(f'Attribute {a} set to {value}')
       