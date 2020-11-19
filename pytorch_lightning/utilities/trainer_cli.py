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
from typing import Type
from jsonargparse import ArgumentParser, ActionConfigFile
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.callbacks import Callback


class LightningArgumentParser(ArgumentParser):
    """Extension of jsonargparse's ArgumentParser for pytorch-lightning"""

    def __init__(
        self,
        *args,
        parse_as_dict: bool = True,
        **kwargs
    ):
        """Initialize argument parser that supports configuration file input"""
        super().__init__(*args, parse_as_dict=parse_as_dict, **kwargs)
        self.add_argument(
            '--config',
            action=ActionConfigFile,
            help='Path to a configuration file in json or yaml format.'
        )

    def add_trainer_args(
        self,
        trainer_class: Type[Trainer] = Trainer,
        nested_key: str = 'trainer'
    ):
        """
        Adds arguments from a trainer class to a nested key of the parser

        Args:
            trainer_class: Optional extension of the Trainer class.
            nested_key: Name of the nested namespace to store arguments.
        """
        assert issubclass(trainer_class, Trainer)
        self.add_class_arguments(trainer_class, nested_key)

    def add_module_args(
        self,
        module_class: Type[LightningModule],
        nested_key: str = 'module'
    ):
        """
        Adds arguments from a module class to a nested key of the parser

        Args:
            module_class: A LightningModule class.
            nested_key: Name of the nested namespace to store arguments.
        """
        assert issubclass(module_class, LightningModule)
        self.add_class_arguments(module_class, nested_key)

    def add_datamodule_args(
        self,
        datamodule_class: Type[LightningDataModule],
        nested_key: str = 'data'
    ):
        """
        Adds arguments from a datamodule class to a nested key of the parser

        Args:
            datamodule_class: A LightningDataModule class.
            nested_key: Name of the nested namespace to store arguments.
        """
        assert issubclass(datamodule_class, LightningDataModule)
        self.add_class_arguments(datamodule_class, nested_key)


class SaveConfigCallback(Callback):
    """Saves a LightningCLI config to the log_dir when training starts"""

    def __init__(self, parser, config):
        self.config_dump = parser.dump(config, skip_none=False)

    def on_train_start(self, trainer, pl_module):
        config_path = os.path.join(trainer.logger.log_dir, 'config.yaml')
        with open(config_path, 'w') as outstream:
            outstream.write(self.config_dump)


class LightningCLI:
    def __init__(
        self,
        model_class: Type[LightningModule],
        datamodule_class: Type[LightningDataModule] = None,
        save_config_callback: Type[Callback] = SaveConfigCallback,
        trainer_class: Type[Trainer] = Trainer,
        description: str = 'pytorch-lightning trainer command line tool',
        parse_env: bool = False,
        **kwargs
    ):
        """
        Implementation of a simple configurable Trainer command line tool

        Receives as input pytorch-lightning classes, which are instantiated using a
        parsed configuration file or command line args and then runs trainer.fit.

        Example, first implement the trainer.py tool as::

            from mymodels import MyModel
            from pytorch_lightning.utilities.jsonargparse_utils import LightningCLI
            LightningCLI(MyModel)

        Then in a shell, run the tool with the desired configuration::

            $ python trainer.py --print-config > config.yaml
            $ nano config.yaml  # modify the config as desired
            $ python trainer.py --cfg config.yaml

        Args:
            model_class: The LightningModule class to train on.
            datamodule_class: An optional LightningDataModule class.
            save_config_callback: A callback class to save the training config.
            trainer_class: An optional extension of the Trainer class.
            description: Description of the tool shown when running --help.
            parse_env: Whether environment variables are also parsed.
            **kwargs: Additional arguments to instantiate Trainer.
        """
        if 'callbacks' not in kwargs:
            kwargs['callbacks'] = []

        self.model_class = model_class
        self.datamodule_class = datamodule_class
        self.save_config_callback = save_config_callback
        self.trainer_class = trainer_class
        self.trainer_kwargs = kwargs

        self.init_parser(description, parse_env)
        self.add_arguments_to_parser(self.parser)
        self.add_core_arguments_to_parser()
        self.parse_arguments()
        self.instantiate_classes()
        self.run()

    def init_parser(
        self,
        description: str,
        parse_env: bool
    ):
        """Method that instantiates the argument parser

        Args:
            description: Description of the tool shown when running --help.
            parse_env: Whether environment variables are also parsed.
        """
        self.parser = LightningArgumentParser(
            description=description,
            print_config='--print_config',
            default_env=parse_env,
            env_prefix='PL'
        )

    def add_arguments_to_parser(
        self,
        parser: LightningArgumentParser
    ):
        """Implement to add extra arguments to parser

        Args:
            parser: The argument parser object to which arguments should be added
        """
        pass

    def add_core_arguments_to_parser(self):
        """Adds arguments from the core classes to the parser"""
        self.parser.add_trainer_args(self.trainer_class, 'trainer')
        self.parser.add_module_args(self.model_class, 'model')
        if self.datamodule_class is not None:
            self.parser.add_datamodule_args(self.datamodule_class, 'data')

    def parse_arguments(self):
        """Parses command line arguments and stores it in self.config"""
        self.config = self.parser.parse_args()

    def instantiate_classes(self):
        """Instantiates the classes using settings from self.config and prepares fit_kwargs"""
        # Instantiate model
        self.model = self.model_class(**self.config.get('model', {}))
        # Instantiate datamodule
        self.fit_kwargs = {'model': self.model}
        if self.datamodule_class is not None:
            self.fit_kwargs['datamodule'] = self.datamodule_class(**self.config.get('data', {}))
        # Instantiate trainer
        self.trainer_kwargs.update(self.config['trainer'])
        if self.save_config_callback is not None:
            self.trainer_kwargs['callbacks'].append(self.save_config_callback(self.parser, self.config))
        self.trainer = self.trainer_class(**self.trainer_kwargs)

    def before_fit(self):
        """Implement to run some code before fit is started"""
        pass

    def after_fit(self):
        """Implement to run some code after fit has finished"""
        pass

    def fit(self):
        """Runs fit of the instantiated trainer class and prepared fit keyword arguments"""
        self.trainer.fit(**self.fit_kwargs)

    def run(self):
        """Runs self.before_fit, then self.fit and finally self.after_fit"""
        self.before_fit()
        self.fit()
        self.after_fit()
