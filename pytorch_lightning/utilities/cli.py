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
from typing import Any, Dict, Optional, Type, Union

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.utilities import _module_available
from pytorch_lightning.utilities.seed import seed_everything

_JSONARGPARSE_AVAILABLE = _module_available("jsonargparse")
if _JSONARGPARSE_AVAILABLE:
    from jsonargparse import ActionConfigFile, ArgumentParser, set_config_read_mode
    set_config_read_mode(fsspec_enabled=True)
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
        config_filename: str,
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
        save_config_filename: str = 'config.yaml',
        trainer_class: Type[Trainer] = Trainer,
        trainer_defaults: Dict[str, Any] = None,
        seed_everything_default: int = None,
        description: str = 'pytorch-lightning trainer command line tool',
        env_prefix: str = 'PL',
        env_parse: bool = False,
        parser_kwargs: Dict[str, Any] = None,
        subclass_mode_model: bool = False,
        subclass_mode_data: bool = False
    ) -> None:
        """
        Receives as input pytorch-lightning classes, which are instantiated
        using a parsed configuration file and/or command line args and then runs
        trainer.fit. Parsing of configuration from environment variables can
        be enabled by setting ``env_parse=True``. A full configuration yaml would
        be parsed from ``PL_CONFIG`` if set. Individual settings are so parsed from
        variables named for example ``PL_TRAINER__MAX_EPOCHS``.

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
        self.save_config_filename = save_config_filename
        self.trainer_class = trainer_class
        self.trainer_defaults = {} if trainer_defaults is None else trainer_defaults
        self.seed_everything_default = seed_everything_default
        self.subclass_mode_model = subclass_mode_model
        self.subclass_mode_data = subclass_mode_data
        self.parser_kwargs = {} if parser_kwargs is None else parser_kwargs
        self.parser_kwargs.update({'description': description, 'env_prefix': env_prefix, 'default_env': env_parse})

        self.init_parser()
        self.add_core_arguments_to_parser()
        self.add_arguments_to_parser(self.parser)
        self.parse_arguments()
        if self.config['seed_everything'] is not None:
            seed_everything(self.config['seed_everything'], workers=True)
        self.before_instantiate_classes()
        self.instantiate_classes()
        self.prepare_fit_kwargs()
        self.before_fit()
        self.fit()
        self.after_fit()

    def init_parser(self) -> None:
        """Method that instantiates the argument parser"""
        self.parser = LightningArgumentParser(**self.parser_kwargs)

    def add_core_arguments_to_parser(self) -> None:
        """Adds arguments from the core classes to the parser"""
        self.parser.add_argument(
            '--seed_everything',
            type=Optional[int],
            default=self.seed_everything_default,
            help='Set to an int to run seed_everything with this value before classes instantiation',
        )
        self.parser.add_lightning_class_args(self.trainer_class, 'trainer')
        trainer_defaults = {'trainer.' + k: v for k, v in self.trainer_defaults.items() if k != 'callbacks'}
        self.parser.set_defaults(trainer_defaults)
        self.parser.add_lightning_class_args(self.model_class, 'model', subclass_mode=self.subclass_mode_model)
        if self.datamodule_class is not None:
            self.parser.add_lightning_class_args(self.datamodule_class, 'data', subclass_mode=self.subclass_mode_data)

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Implement to add extra arguments to parser or link arguments

        Args:
            parser: The argument parser object to which arguments can be added
        """

    def parse_arguments(self) -> None:
        """Parses command line arguments and stores it in self.config"""
        self.config = self.parser.parse_args()

    def before_instantiate_classes(self) -> None:
        """Implement to run some code before instantiating the classes"""

    def instantiate_classes(self) -> None:
        """Instantiates the classes using settings from self.config"""
        self.config_init = self.parser.instantiate_subclasses(self.config)
        self.instantiate_datamodule()
        self.instantiate_model()
        self.instantiate_trainer()

    def instantiate_datamodule(self) -> None:
        """Instantiates the datamodule using self.config_init['data'] if given"""
        if self.datamodule_class is None:
            self.datamodule = None
        elif self.subclass_mode_data:
            self.datamodule = self.config_init['data']
        else:
            self.datamodule = self.datamodule_class(**self.config_init.get('data', {}))

    def instantiate_model(self) -> None:
        """Instantiates the model using self.config_init['model']"""
        if self.subclass_mode_model:
            self.model = self.config_init['model']
        else:
            self.model = self.model_class(**self.config_init.get('model', {}))

    def instantiate_trainer(self) -> None:
        """Instantiates the trainer using self.config_init['trainer']"""
        if self.config_init['trainer'].get('callbacks') is None:
            self.config_init['trainer']['callbacks'] = []
        if 'callbacks' in self.trainer_defaults:
            if isinstance(self.trainer_defaults['callbacks'], list):
                self.config_init['trainer']['callbacks'].extend(self.trainer_defaults['callbacks'])
            else:
                self.config_init['trainer']['callbacks'].append(self.trainer_defaults['callbacks'])
        if self.save_config_callback is not None:
            config_callback = self.save_config_callback(self.parser, self.config, self.save_config_filename)
            self.config_init['trainer']['callbacks'].append(config_callback)
        self.trainer = self.trainer_class(**self.config_init['trainer'])

    def prepare_fit_kwargs(self) -> None:
        """Prepares fit_kwargs including datamodule using self.config_init['data'] if given"""
        self.fit_kwargs = {'model': self.model}
        if self.datamodule is not None:
            self.fit_kwargs['datamodule'] = self.datamodule

    def before_fit(self) -> None:
        """Implement to run some code before fit is started"""

    def fit(self) -> None:
        """Runs fit of the instantiated trainer class and prepared fit keyword arguments"""
        self.trainer.fit(**self.fit_kwargs)

    def after_fit(self) -> None:
        """Implement to run some code after fit has finished"""
