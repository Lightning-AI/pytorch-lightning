from typing import Union, Optional, List, Tuple, Any
from argparse import ArgumentParser, Namespace
import inspect

from torch.utils.data import DataLoader

from pytorch_lightning import _logger as log
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.hypertuner.lr_finder import HyperTunerLRFinderMixin
from pytorch_lightning.hypertuner.batch_scaler import HyperTunerBatchScalerMixin
from pytorch_lightning.hypertuner.n_worker_search import HyperTunerNworkerSearchMixin
from pytorch_lightning.utilities.parsing import lightning_hasattr, lightning_setattr
from pytorch_lightning.utilities.exceptions import MisconfigurationException


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
            self._call_internally(self.scale_batch_size,
                                  self.auto_scale_batch_size,
                                  'batch_size')

        # N worker search
        if self.auto_n_worker_search:
            self._call_internally(self.n_worker_search,
                                  self.auto_n_worker_search,
                                  'n_workers')

        # Lr finder
        if self.auto_lr_find:
            self._call_internally(self.lr_find,
                                  self.auto_lr_find,
                                  'learning_rate')

    def _call_internally(self, model, method, attribute, default):
        attribute = attribute if isinstance(attribute, str) else default

        # Check that user has the wanted attribute in their model
        if not lightning_hasattr(model, attribute):
            raise MisconfigurationException('model or model.hparams does not have'
                                            f' a field called {attribute} which is'
                                            f' required by tuner algorithm {method}')

        # Call method
        obj = method(model, attribute_name=attribute)

        # Get suggested value
        value = obj.suggestion()

        # Set value in model
        lightning_setattr(model, attribute, value)
        log.info(f'Tuner method {method} completed. Attribute {attribute} set to {value}.')

    @classmethod
    def get_init_arguments_and_types(cls) -> List[Tuple[str, Tuple, Any]]:
        """ Returns a list with tuples of 3 values:
            (argument name, set with argument types, argument default value).
        """
        return [('auto_scale_batch_size', (bool, str), False),
                ('auto_lr_find', (bool, str), False),
                ('auto_n_worker_search', (bool, str), False)]

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser) -> ArgumentParser:
        r"""Extends existing argparse by default `HyperTuner` attributes.

        Args:
            parent_parser:
                The custom cli arguments parser, which will be extended by
                the Trainer default arguments.

        Examples:
            >>> import argparse
            >>> import pprint
            >>> parser = argparse.ArgumentParser()
            >>> parser = HyperTuner.add_argparse_args(parser)
            >>> args = parser.parse_args([])
            >>> pprint.pprint(vars(args))
            {'auto_scale_batch_size': False,
             'auto_lr_find': False,
             'auto_n_worker_search': False}

        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False,)

        for (arg, arg_types, arg_default) in cls.get_init_arguments_and_types():

            arg_kwargs = {}
            arg_kwargs.update(nargs="?")
            use_type = [at for at in arg_types if at is not bool][0]

            parser.add_argument(
                f'--{arg}',
                dest=arg,
                default=arg_default,
                type=use_type,
                help='autogenerated by pl.Trainer',
                **arg_kwargs,
            )

        return parser

    @classmethod
    def parse_argparser(cls, arg_parser: Union[ArgumentParser, Namespace]) -> Namespace:
        """Parse CLI arguments, required for custom bool types."""
        args = arg_parser.parse_args() if isinstance(arg_parser, ArgumentParser) else arg_parser

        types_default = {
            arg: (arg_types, arg_default) for arg, arg_types, arg_default in cls.get_init_arguments_and_types()
        }

        modified_args = {}
        for k, v in vars(args).items():
            if k in types_default and v is None:
                arg_types, arg_default = types_default[k]
                if bool in arg_types and isinstance(arg_default, bool):
                    # Value has been passed as a flag => It is currently None, so we need to set it to True
                    # We always set to True, regardless of the default value.
                    # Users must pass False directly, but when passing nothing True is assumed.
                    # i.e. the only way to disable somthing that defaults to True is to use the long form:
                    # "--a_default_true_arg False" becomes False, while "--a_default_false_arg" becomes None,
                    # which then becomes True here.
                    v = True

            modified_args[k] = v
        return Namespace(**modified_args)

    @classmethod
    def from_argparse_args(cls, args: Union[Namespace, ArgumentParser], **kwargs) -> 'HyperTuner':
        """
        Create an instance from CLI arguments.

        Args:
            args: The parser or namespace to take arguments from. Only known arguments will be
                parsed and passed to the :class:`Trainer`.
            **kwargs: Additional keyword arguments that may override ones in the parser or namespace.
                These must be valid Trainer arguments.

        Example:
            >>> model = LightningModule(...)
            >>> trainer = Trainer(...)
            >>> parser = ArgumentParser(add_help=False)
            >>> parser = HyperTuner.add_argparse_args(parser)
            >>> args = HyperTuner.parse_argparser(parser.parse_args(""))
            >>> tuner = HyperTuner.from_argparse_args(args, trainer=trainer, model=model)
        """
        if isinstance(args, ArgumentParser):
            args = cls.parse_argparser(args)
        params = vars(args)

        valid_kwargs = inspect.signature(cls.__init__).parameters
        trainer_kwargs = dict((name, params[name]) for name in valid_kwargs if name in params)
        trainer_kwargs.update(**kwargs)

        return cls(**trainer_kwargs)
