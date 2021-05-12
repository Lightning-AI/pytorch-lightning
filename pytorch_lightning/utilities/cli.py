# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from argparse import Namespace
from typing import Any, Dict, List, Optional, Type, Union

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.utilities import _module_available
from pytorch_lightning.utilities.seed import seed_everything

_JSONARGPARSE_AVAILABLE = _module_available("jsonargparse")
if _JSONARGPARSE_AVAILABLE:
    from jsonargparse import ActionConfigFile, ArgumentParser
else:
    ArgumentParser = object


class LightningArgumentParser(ArgumentParser):
    """Extension of jsonargparse's ArgumentParser for pytorch-lightning"""

    def __init__(self, *args, parse_as_dict: bool = True, **kwargs) -> None:
        """Initialize argument parser that supports configuration file input

        For full details of accepted arguments see `ArgumentParser.__init__
        <https://jsonargparse.readthedocs.io/en/stable/#jsonargparse.core.ArgumentParser.__init__>`_.
        """
        if not _JSONARGPARSE_AVAILABLE:
            raise ModuleNotFoundError(
                '`jsonargparse` is not installed but it is required for the CLI.'
                ' Install it with `pip install jsonargparse[signatures]`.'
            )
        super().__init__(*args, parse_as_dict=parse_as_dict, **kwargs)
        self.add_argument(
            '--config', action=ActionConfigFile, help='Path to a configuration file in json or yaml format.'
        )

    def add_lightning_class_args(
        self,
        lightning_class: Union[Type[Trainer], Type[LightningModule], Type[LightningDataModule]],
        nested_key: str,
        subclass_mode: bool = False
    ) -> None:
        """
        Adds arguments from a lightning class to a nested key of the parser

        Args:
            lightning_class: Any subclass of {Trainer,LightningModule,LightningDataModule}.
            nested_key: Name of the nested namespace to store arguments.
            subclass_mode: Whether allow any subclass of the given class.
        """
        assert issubclass(lightning_class, (Trainer, LightningModule, LightningDataModule))
        if subclass_mode:
            return self.add_subclass_arguments(lightning_class, nested_key, required=True)
        return self.add_class_arguments(lightning_class, nested_key, fail_untyped=False)


class SaveConfigCallback(Callback):
    """Saves a LightningCLI config to the log_dir when training starts"""

    def __init__(
        self,
        parser: LightningArgumentParser,
        config: Union[Namespace, Dict[str, Any]],
        config_filename: str = 'config.yaml'
    ) -> None:
        self.parser = parser
        self.config = config
        self.config_filename = config_filename

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        log_dir = trainer.log_dir or trainer.default_root_dir
        config_path = os.path.join(log_dir, self.config_filename)
        self.parser.save(self.config, config_path, skip_none=False)


class LightningCLI:
    """Implementation of a configurable command line tool for pytorch-lightning"""

    def __init__(
        self,
        model_class: Type[LightningModule],
        datamodule_class: Type[LightningDataModule] = None,
        save_config_callback: Type[SaveConfigCallback] = SaveConfigCallback,
        trainer_class: Type[Trainer] = Trainer,
        trainer_defaults: Dict[str, Any] = None,
        #trainer_fn: str = TrainerFn.fit.value,
        seed_everything_default: int = None,
        description: str = 'pytorch-lightning trainer command line tool',
        env_prefix: str = 'PL',
        env_parse: bool = False,
        parser_kwargs: Dict[str, Any] = None,
        subclass_mode_model: bool = False,
        subclass_mode_data: bool = False
    ) -> None:
        """
        Receives as input pytorch-lightning classes, which are instantiated using
        a parsed configuration file and/or command line args and then runs ``trainer.fit()`` (by default).
        Parsing of configuration from environment variables can be enabled by setting ``env_parse=True``.
        A full configuration yaml would be parsed from ``PL_CONFIG`` if set.
        Individual settings are so parsed from variables named for example ``PL_TRAINER__MAX_EPOCHS``.

        Example, first implement the ``trainer.py`` tool as::

            from mymodels import MyModel
            from pytorch_lightning.utilities.cli import LightningCLI
            LightningCLI(MyModel)

        Then in a shell, run the tool with the desired configuration::

            $ python trainer.py --print_config > config.yaml
            $ nano config.yaml  # modify the config as desired
            $ python trainer.py --cfg config.yaml

        .. warning:: ``LightningCLI`` is in beta and subject to change.

        Args:
            model_class: :class:`~pytorch_lightning.core.lightning.LightningModule` class to train on.
            datamodule_class: An optional :class:`~pytorch_lightning.core.datamodule.LightningDataModule` class.
            save_config_callback: A callback class to save the training config.
            trainer_class: An optional subclass of the :class:`~pytorch_lightning.trainer.trainer.Trainer` class.
            trainer_defaults: Set to override Trainer defaults or add persistent callbacks.
            trainer_fn: The trainer function to run.
            seed_everything_default: Default value for the :func:`~pytorch_lightning.utilities.seed.seed_everything`
                seed argument.
            description: Description of the tool shown when running ``--help``.
            env_prefix: Prefix for environment variables.
            env_parse: Whether environment variable parsing is enabled.
            parser_kwargs: Additional arguments to instantiate LightningArgumentParser.
            subclass_mode_model: Whether model can be any `subclass
                <https://jsonargparse.readthedocs.io/en/stable/#class-type-and-sub-classes>`_
                of the given class.
            subclass_mode_data: Whether datamodule can be any `subclass
                <https://jsonargparse.readthedocs.io/en/stable/#class-type-and-sub-classes>`_
                of the given class.
        """
        assert issubclass(trainer_class, Trainer)
        assert issubclass(model_class, LightningModule)
        if datamodule_class is not None:
            assert issubclass(datamodule_class, LightningDataModule)
        self.model_class = model_class
        self.datamodule_class = datamodule_class
        self.save_config_callback = save_config_callback
        self.trainer_class = trainer_class
        self.trainer_defaults = trainer_defaults or {}
        #self.trainer_fn = trainer_fn
        self.seed_everything_default = seed_everything_default
        self.subclass_mode_model = subclass_mode_model
        self.subclass_mode_data = subclass_mode_data

        parser_kwargs = parser_kwargs or {}
        parser_kwargs.update({'description': description, 'env_prefix': env_prefix, 'default_env': env_parse})
        self.parser = self.init_parser(**parser_kwargs)

        self.add_core_arguments(self.parser)
        self.add_arguments(self.parser)
        self.add_subcommands(self.parser)
        self.parse_arguments(self.parser)

        if self.config['seed_everything'] is not None:
            seed_everything(self.config['seed_everything'], workers=True)

        self.before_instantiate_classes()
        self.instantiate_classes()

        self.run_subcommand()

    def init_parser(self, use_base: bool = False, **kwargs: Any) -> LightningArgumentParser:
        """Method that instantiates the argument parser"""
        cls = ArgumentParser if use_base else LightningArgumentParser
        return cls(**kwargs)

    def add_arguments(self, parser: LightningArgumentParser) -> None:
        """
        Implement to add extra arguments to the base parser or link arguments

        Args:
            parser: The base parser object to which arguments can be added
        """
        parser.add_argument(
            '--seed_everything',
            type=Optional[int],
            default=self.seed_everything_default,
            help='Set to an int to run seed_everything with this value before classes instantiation',
        )

    @property
    def subcommands(self) -> Dict[str, List[str]]:
        """Defines the list of available subcommands and the arguments to skip"""
        return {
            'fit': ['model', 'train_dataloader', 'val_dataloaders', 'datamodule'],
            'validate': ['model', 'val_dataloaders', 'datamodule'],
            'test': ['model', 'test_dataloaders', 'datamodule'],
            'predict': ['model', 'dataloaders', 'datamodule'],
            'tune': ['model', 'train_dataloader', 'val_dataloaders', 'datamodule'],
        }

    def add_subcommands(self, parser: LightningArgumentParser) -> None:
        # TODO: default fit
        parser_subcommands = parser.add_subcommands()
        for subcommand in self.subcommands:
            subcommand_parser = self.prepare_subcommand_parser(subcommand)
            # TODO: add help
            parser_subcommands.add_subcommand(subcommand, subcommand_parser)

    def prepare_subcommand_parser(self, subcommand: str) -> LightningArgumentParser:
        # TODO: pass env_prefix and default_env?
        parser = self.init_parser(use_base=True)
        self.add_subcommand_arguments(parser)
        skip = self.subcommands[subcommand]
        parser.add_method_arguments(self.trainer_class, subcommand, skip=skip)
        return parser

    def add_subcommand_arguments(self, parser: LightningArgumentParser) -> None:
        """
        Implement to add extra arguments to each subcommand parser or link arguments

        Args:
            parser: The subcommand parser object to which arguments can be added
        """

    def add_core_arguments(self, parser: LightningArgumentParser) -> None:
        """Adds arguments from the core classes to the parser"""
        parser.add_lightning_class_args(self.trainer_class, 'trainer')
        trainer_defaults = {'trainer.' + k: v for k, v in self.trainer_defaults.items() if k != 'callbacks'}
        parser.set_defaults(trainer_defaults)
        parser.add_lightning_class_args(self.model_class, 'model', subclass_mode=self.subclass_mode_model)
        if self.datamodule_class is not None:
            parser.add_lightning_class_args(self.datamodule_class, 'data', subclass_mode=self.subclass_mode_data)

    def parse_arguments(self, parser: LightningArgumentParser) -> None:
        """Parses command line arguments and stores it in self.config"""
        self.config = parser.parse_args()

    def before_instantiate_classes(self) -> None:
        """Implement to run some code before instantiating the classes"""

    def instantiate_classes(self) -> None:
        """Instantiates the classes using settings from self.config"""
        self.config_init = self.parser.instantiate_subclasses(self.config)
        self.instantiate_datamodule(self.config_init.get('data', {}))
        self.instantiate_model(self.config_init.get('model', {}))
        self.instantiate_trainer(self.config_init['trainer'])

    def instantiate_datamodule(self, config: Dict[str, Any]) -> None:
        """Instantiates the datamodule using self.config_init['data'] if given"""
        if self.datamodule_class is None:
            self.datamodule = None
        elif self.subclass_mode_data:
            self.datamodule = config
        else:
            self.datamodule = self.datamodule_class(**config)

    def instantiate_model(self, config: Dict[str, Any]) -> None:
        """Instantiates the model using self.config_init['model']"""
        if self.subclass_mode_model:
            self.model = config
        else:
            self.model = self.model_class(**config)

    def instantiate_trainer(self, config: Dict[str, Any]) -> None:
        """Instantiates the trainer using self.config_init['trainer']"""
        if config.get('callbacks') is None:
            config['callbacks'] = []
        if 'callbacks' in self.trainer_defaults:
            if isinstance(self.trainer_defaults['callbacks'], list):
                config['callbacks'].extend(self.trainer_defaults['callbacks'])
            else:
                config['callbacks'].append(self.trainer_defaults['callbacks'])
        if self.save_config_callback is not None:
            config['callbacks'].append(self.save_config_callback(self.parser, self.config))
        self.trainer = self.trainer_class(**config)

    def run_subcommand(self) -> None:
        """Run the chosen subcommand"""
        subcommand = self.config['subcommand']
        fn_kwargs = self.prepare_subcommand_kwargs(subcommand)

        if hasattr(self, f'before_{subcommand}'):
            getattr(self, f'before_{subcommand}')()

        default = getattr(self.trainer, subcommand)
        fn = getattr(self, subcommand, default)
        fn(**fn_kwargs)

        if hasattr(self, f'after_{subcommand}'):
            getattr(self, f'after_{subcommand}')()

    def prepare_subcommand_kwargs(self, subcommand: str) -> Dict[str, Any]:
        """Prepares the keyword arguments to pass to the subcommand to run"""
        fn_kwargs = {'model': self.model, **self.config_init[subcommand]}
        if self.datamodule is not None:
            fn_kwargs['datamodule'] = self.datamodule
        return fn_kwargs
