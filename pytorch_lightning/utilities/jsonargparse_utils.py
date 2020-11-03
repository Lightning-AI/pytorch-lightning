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
from pytorch_lightning import Trainer, LightningModule, LightningDataModule
from pytorch_lightning.callbacks import Callback


class LightningArgumentParser(ArgumentParser):
    def __init__(self, *args, **kwargs):
        """Initialize argument parser that supports configuration file input"""
        super().__init__(*args, **kwargs)
        self.add_argument('--cfg',
            action=ActionConfigFile,
            help='Path to a configuration file in json or yaml format.')


    def add_trainer_args(
        self,
        trainer_class: Type[Trainer] = Trainer,
        nested_key: str = 'trainer'
    ):
        """
        Adds arguments from a trainer class to a nested key of the parser

        Args:
            trainer_class: Optional extension of the Trainer class.
            nested_key: Name of the nested namespace where parsed arguments are stored.
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
            nested_key: Name of the nested namespace where parsed arguments are stored.
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
            nested_key: Name of the nested namespace where parsed arguments are stored.
        """
        assert issubclass(datamodule_class, LightningDataModule)
        self.add_class_arguments(datamodule_class, nested_key)


class SaveConfigCallback(Callback):
    """Callback that saves a trainer_cli config to the log_dir when training starts"""

    def __init__(self, parser, config):
        self.config_dump = parser.dump(config, skip_none=False)


    def on_train_start(self, trainer, pl_module):
        config_path = os.path.join(trainer.logger.log_dir, 'config.yaml')
        with open(config_path, 'w') as outstream:
            outstream.write(self.config_dump)


def trainer_cli(
    model_class: Type[LightningModule],
    datamodule_class: Type[LightningDataModule] = None,
    save_config_callback: Type[Callback] = SaveConfigCallback,
    trainer_class: Type[Trainer] = Trainer,
    description: str = 'pytorch-lightning trainer command line tool',
    parse_env: bool = False,
):
    """
    Implementation of a simple configurable Trainer command line tool

    Receives as input pytorch-lightning classes, which are instantiated using a
    parsed configuration file or command line options and then runs trainer.fit.

    Example, first implement the trainer.py tool as::

        from mymodels import MyModel
        from pytorch_lightning.utilities.jsonargparse_utils import trainer_cli
        trainer_cli(MyModel)

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
    """
    # Define parser
    parser = LightningArgumentParser(description=description,
                                     parse_as_dict=True,
                                     default_env=parse_env,
                                     env_prefix='PL')
    parser.add_trainer_args(trainer_class, 'trainer')
    parser.add_module_args(model_class, 'model')
    if datamodule_class is not None:
        parser.add_datamodule_args(datamodule_class, 'data')

    # Parse configuration
    config = parser.parse_args()

    # Instantiate classes
    model = model_class(**config.get('model', {}))
    kwargs = {'model': model}
    if datamodule_class is not None:
        kwargs['datamodule'] = datamodule_class(**config.get('data', {}))

    if save_config_callback is not None:
        config['trainer']['callbacks'] = [save_config_callback(parser, config)]
    trainer = Trainer(**config['trainer'])

    # Start training
    trainer.fit(**kwargs)
