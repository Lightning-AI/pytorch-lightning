from typing import Union, Optional, List

from torch.utils.data import DataLoader

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.hypertuner.lr_finder import HyperTunerLRFinderMixin
from pytorch_lightning.hypertuner.batch_scaler import HyperTunerBatchScalerMixin
from pytorch_lightning.hypertuner.n_worker_search import HyperTunerNworkerSearchMixin


class HyperTuner(HyperTunerLRFinderMixin,
                 HyperTunerBatchScalerMixin,
                 HyperTunerNworkerSearchMixin):
    r"""
    HyperTuner class can help tuning hyperparameters before fitting your model.
    This is not a general purpose hyperparameter optimization class but it uses
    deterministic methods for tuning certain hyperparameters in your training
    related to speed and convergence.

    Currently the class support tuning the learning rate, batch size and
    number of workers of your model.

    Args:
        trainer: instance of pl.Trainer

        model: instance of pl.LightningModule

        auto_lr_find: If set to True, will run a learning rate finder,
            trying to optimize initial learning for faster convergence. Automatically
            adjust either `model.lr`, `model.hparams.lr`.
            To use a different key, set a string instead of True with the field name.

        auto_scale_batch_size: If set to True, will run a batch size scaler
            trying to find the largest batch size that fits into memory. Automatically
            adjust either `model.batch_size` or `model.hparams.batch_size`
            To use a different key, set a string instead of True with the field name.

        auto_n_worker_search: If set to True, will run a n-worker search algortihm
            that tries to find the optimal number of workers to use for your dataloaders.
            Automatically adjust either `model.n_workers` or `model.hparams.n_workers`
    """

    # set methods that should be called AFTER, i.e. scale batch size should be
    # called before lr_find and n_worker_search. This is a general setup for
    # future method
    call_order = {'scale_batch_size': ['lr_find', 'n_worker_search'],
                  'lr_find': [],
                  'n_worker_search': []}

    def __init__(self,
                 trainer: Trainer,
                 model: LightningModule,
                 auto_scale_batch_size: Union[str, bool] = False,
                 auto_lr_find: Union[bool, str] = False,
                 auto_n_worker_search: Union[bool, str] = False):

        # User instance of trainer and model
        self.trainer = trainer
        self.model = model

        # Parameters to optimize
        self.auto_scale_batch_size = auto_scale_batch_size
        self.auto_lr_find = auto_lr_find
        self.auto_n_worker_search = auto_n_worker_search

        # For checking dependency
        self._scale_batch_size_called = False
        self._lr_find_called = False
        self._n_worker_search_called = False

    def tune(self,
             train_dataloader: Optional[DataLoader] = None,
             val_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
             datamodule: Optional[LightningDataModule] = None,
             ):
        r"""
        Automatic run the enabled tuner algorithms

        Args:
            train_dataloader: A Pytorch
                DataLoader with training samples. If the model has
                a predefined train_dataloader method this will be skipped.

            val_dataloaders: Either a single
                Pytorch Dataloader or a list of them, specifying validation samples.
                If the model has a predefined val_dataloaders method this will be skipped

            datamodule: instance of type pl.DataModule. You cannot pass train_dataloader
                or val_dataloaders to HyperTuner.tune if you supply a datamodule

        Example::
            # Automatically tune hyperparameters
            from pytorch_lightning import Trainer, HyperTuner
            model = ModelClass(...)
            trainer = Trainer(...)
            tuner = HyperTuner(trainer, model
                               auto_scale_batch_size=True,
                               auto_lr_find=True,
                               auto_n_worker_search=True)
            tuner.tune()  # automatically tunes hyperparameters

            # Do standard training with optimized parameters
            trainer.fit(model)
        """
        # Batch size scaling
        if self.auto_scale_batch_size:
            self.scale_batch_size()

        # N worker search
        if self.auto_n_worker_search:
            self.n_worker_search()

        # Lr finder
        if self.auto_lr_find:
            self.lr_find()
