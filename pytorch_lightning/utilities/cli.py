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
"""Utilities for LightningCLI."""

import inspect
import os
import sys
from functools import partial, update_wrapper
from types import MethodType, ModuleType
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Type, Union
from unittest import mock

import torch
import yaml
from torch.optim import Optimizer

import pytorch_lightning as pl
from pytorch_lightning import Callback, LightningDataModule, LightningModule, seed_everything, Trainer
from pytorch_lightning.utilities.cloud_io import get_filesystem
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _DOCSTRING_PARSER_AVAILABLE, _JSONARGPARSE_AVAILABLE
from pytorch_lightning.utilities.meta import get_all_subclasses
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import _warn, rank_zero_warn
from pytorch_lightning.utilities.types import LRSchedulerType, LRSchedulerTypeTuple, LRSchedulerTypeUnion

if _JSONARGPARSE_AVAILABLE:
    from jsonargparse import ActionConfigFile, ArgumentParser, class_from_function, Namespace, set_config_read_mode

    set_config_read_mode(fsspec_enabled=True)
else:
    locals()["ArgumentParser"] = object
    locals()["Namespace"] = object

if _DOCSTRING_PARSER_AVAILABLE:
    import docstring_parser


class _Registry(dict):
    def __call__(self, cls: Type, key: Optional[str] = None, override: bool = False) -> Type:
        """Registers a class mapped to a name.

        Args:
            cls: the class to be mapped.
            key: the name that identifies the provided class.
            override: Whether to override an existing key.
        """
        if key is None:
            key = cls.__name__
        elif not isinstance(key, str):
            raise TypeError(f"`key` must be a str, found {key}")

        if key not in self or override:
            self[key] = cls
        return cls

    def register_classes(self, module: ModuleType, base_cls: Type, override: bool = False) -> None:
        """This function is an utility to register all classes from a module."""
        for cls in self.get_members(module, base_cls):
            self(cls=cls, override=override)

    @staticmethod
    def get_members(module: ModuleType, base_cls: Type) -> Generator[Type, None, None]:
        return (
            cls
            for _, cls in inspect.getmembers(module, predicate=inspect.isclass)
            if issubclass(cls, base_cls) and cls != base_cls
        )

    @property
    def names(self) -> List[str]:
        """Returns the registered names."""
        return list(self.keys())

    @property
    def classes(self) -> Tuple[Type, ...]:
        """Returns the registered classes."""
        return tuple(self.values())

    def __str__(self) -> str:
        return f"Registered objects: {self.names}"


OPTIMIZER_REGISTRY = _Registry()
LR_SCHEDULER_REGISTRY = _Registry()
CALLBACK_REGISTRY = _Registry()
MODEL_REGISTRY = _Registry()
DATAMODULE_REGISTRY = _Registry()
LOGGER_REGISTRY = _Registry()


class ReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, optimizer: Optimizer, monitor: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(optimizer, *args, **kwargs)
        self.monitor = monitor


def _populate_registries(subclasses: bool) -> None:
    if subclasses:
        # this will register any subclasses from all loaded modules including userland
        for cls in get_all_subclasses(torch.optim.Optimizer):
            OPTIMIZER_REGISTRY(cls)
        for cls in get_all_subclasses(torch.optim.lr_scheduler._LRScheduler):
            LR_SCHEDULER_REGISTRY(cls)
        for cls in get_all_subclasses(pl.Callback):
            CALLBACK_REGISTRY(cls)
        for cls in get_all_subclasses(pl.LightningModule):
            MODEL_REGISTRY(cls)
        for cls in get_all_subclasses(pl.LightningDataModule):
            DATAMODULE_REGISTRY(cls)
        for cls in get_all_subclasses(pl.loggers.LightningLoggerBase):
            LOGGER_REGISTRY(cls)
    else:
        # manually register torch's subclasses and our subclasses
        OPTIMIZER_REGISTRY.register_classes(torch.optim, Optimizer)
        LR_SCHEDULER_REGISTRY.register_classes(torch.optim.lr_scheduler, torch.optim.lr_scheduler._LRScheduler)
        CALLBACK_REGISTRY.register_classes(pl.callbacks, pl.Callback)
        LOGGER_REGISTRY.register_classes(pl.loggers, pl.loggers.LightningLoggerBase)
    # `ReduceLROnPlateau` does not subclass `_LRScheduler`
    LR_SCHEDULER_REGISTRY(cls=ReduceLROnPlateau)


class LightningArgumentParser(ArgumentParser):
    """Extension of jsonargparse's ArgumentParser for pytorch-lightning."""

    # use class attribute because `parse_args` is only called on the main parser
    _choices: Dict[str, Tuple[Tuple[Type, ...], bool]] = {}

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize argument parser that supports configuration file input.

        For full details of accepted arguments see `ArgumentParser.__init__
        <https://jsonargparse.readthedocs.io/en/stable/index.html#jsonargparse.ArgumentParser.__init__>`_.
        """
        if not _JSONARGPARSE_AVAILABLE:
            raise ModuleNotFoundError(
                "`jsonargparse` is not installed but it is required for the CLI."
                " Install it with `pip install -U jsonargparse[signatures]`."
            )
        super().__init__(*args, **kwargs)
        self.add_argument(
            "-c", "--config", action=ActionConfigFile, help="Path to a configuration file in json or yaml format."
        )
        self.callback_keys: List[str] = []
        # separate optimizers and lr schedulers to know which were added
        self._optimizers: Dict[str, Tuple[Union[Type, Tuple[Type, ...]], str]] = {}
        self._lr_schedulers: Dict[str, Tuple[Union[Type, Tuple[Type, ...]], str]] = {}

    def add_lightning_class_args(
        self,
        lightning_class: Union[
            Callable[..., Union[Trainer, LightningModule, LightningDataModule, Callback]],
            Type[Trainer],
            Type[LightningModule],
            Type[LightningDataModule],
            Type[Callback],
        ],
        nested_key: str,
        subclass_mode: bool = False,
        required: bool = True,
    ) -> List[str]:
        """Adds arguments from a lightning class to a nested key of the parser.

        Args:
            lightning_class: A callable or any subclass of {Trainer, LightningModule, LightningDataModule, Callback}.
            nested_key: Name of the nested namespace to store arguments.
            subclass_mode: Whether allow any subclass of the given class.
            required: Whether the argument group is required.

        Returns:
            A list with the names of the class arguments added.
        """
        if callable(lightning_class) and not isinstance(lightning_class, type):
            lightning_class = class_from_function(lightning_class)

        if isinstance(lightning_class, type) and issubclass(
            lightning_class, (Trainer, LightningModule, LightningDataModule, Callback)
        ):
            if issubclass(lightning_class, Callback):
                self.callback_keys.append(nested_key)
            if subclass_mode:
                return self.add_subclass_arguments(lightning_class, nested_key, fail_untyped=False, required=required)
            return self.add_class_arguments(
                lightning_class,
                nested_key,
                fail_untyped=False,
                instantiate=not issubclass(lightning_class, Trainer),
                sub_configs=True,
            )
        raise MisconfigurationException(
            f"Cannot add arguments from: {lightning_class}. You should provide either a callable or a subclass of: "
            "Trainer, LightningModule, LightningDataModule, or Callback."
        )

    def add_optimizer_args(
        self,
        optimizer_class: Union[Type[Optimizer], Tuple[Type[Optimizer], ...]],
        nested_key: str = "optimizer",
        link_to: str = "AUTOMATIC",
    ) -> None:
        """Adds arguments from an optimizer class to a nested key of the parser.

        Args:
            optimizer_class: Any subclass of :class:`torch.optim.Optimizer`.
            nested_key: Name of the nested namespace to store arguments.
            link_to: Dot notation of a parser key to set arguments or AUTOMATIC.
        """
        if isinstance(optimizer_class, tuple):
            assert all(issubclass(o, Optimizer) for o in optimizer_class)
        else:
            assert issubclass(optimizer_class, Optimizer)
        kwargs = {"instantiate": False, "fail_untyped": False, "skip": {"params"}}
        if isinstance(optimizer_class, tuple):
            self.add_subclass_arguments(optimizer_class, nested_key, **kwargs)
            self.set_choices(nested_key, optimizer_class)
        else:
            self.add_class_arguments(optimizer_class, nested_key, sub_configs=True, **kwargs)
        self._optimizers[nested_key] = (optimizer_class, link_to)

    def add_lr_scheduler_args(
        self,
        lr_scheduler_class: Union[LRSchedulerType, Tuple[LRSchedulerType, ...]],
        nested_key: str = "lr_scheduler",
        link_to: str = "AUTOMATIC",
    ) -> None:
        """Adds arguments from a learning rate scheduler class to a nested key of the parser.

        Args:
            lr_scheduler_class: Any subclass of ``torch.optim.lr_scheduler.{_LRScheduler, ReduceLROnPlateau}``.
            nested_key: Name of the nested namespace to store arguments.
            link_to: Dot notation of a parser key to set arguments or AUTOMATIC.
        """
        if isinstance(lr_scheduler_class, tuple):
            assert all(issubclass(o, LRSchedulerTypeTuple) for o in lr_scheduler_class)
        else:
            assert issubclass(lr_scheduler_class, LRSchedulerTypeTuple)
        kwargs = {"instantiate": False, "fail_untyped": False, "skip": {"optimizer"}}
        if isinstance(lr_scheduler_class, tuple):
            self.add_subclass_arguments(lr_scheduler_class, nested_key, **kwargs)
            self.set_choices(nested_key, lr_scheduler_class)
        else:
            self.add_class_arguments(lr_scheduler_class, nested_key, sub_configs=True, **kwargs)
        self._lr_schedulers[nested_key] = (lr_scheduler_class, link_to)

    def parse_args(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        argv = sys.argv
        for k, v in self._choices.items():
            if not any(arg.startswith(f"--{k}") for arg in argv):
                # the key wasn't passed - maybe defined in a config, maybe it's optional
                continue
            classes, is_list = v
            # knowing whether the argument is a list type automatically would be too complex
            if is_list:
                argv = self._convert_argv_issue_85(classes, k, argv)
            else:
                argv = self._convert_argv_issue_84(classes, k, argv)
        self._choices.clear()
        with mock.patch("sys.argv", argv):
            return super().parse_args(*args, **kwargs)

    def set_choices(self, nested_key: str, classes: Tuple[Type, ...], is_list: bool = False) -> None:
        """Adds support for shorthand notation for a particular nested key.

        Args:
            nested_key: The key whose choices will be set.
            classes: A tuple of classes to choose from.
            is_list: Whether the argument is a ``List[object]`` type.
        """
        self._choices[nested_key] = (classes, is_list)

    @staticmethod
    def _convert_argv_issue_84(classes: Tuple[Type, ...], nested_key: str, argv: List[str]) -> List[str]:
        """Placeholder for https://github.com/omni-us/jsonargparse/issues/84.

        Adds support for shorthand notation for ``object`` arguments.
        """
        passed_args, clean_argv = {}, []
        argv_key = f"--{nested_key}"
        # get the argv args for this nested key
        i = 0
        while i < len(argv):
            arg = argv[i]
            if arg.startswith(argv_key):
                if "=" in arg:
                    key, value = arg.split("=")
                else:
                    key = arg
                    i += 1
                    value = argv[i]
                passed_args[key] = value
            else:
                clean_argv.append(arg)
            i += 1

        # the user requested a help message
        help_key = argv_key + ".help"
        if help_key in passed_args:
            argv_class = passed_args[help_key]
            if "." in argv_class:
                # user passed the class path directly
                class_path = argv_class
            else:
                # convert shorthand format to the classpath
                for cls in classes:
                    if cls.__name__ == argv_class:
                        class_path = _class_path_from_class(cls)
                        break
                else:
                    raise ValueError(f"Could not generate get the class_path for {repr(argv_class)}")
            return clean_argv + [help_key, class_path]

        # generate the associated config file
        argv_class = passed_args.pop(argv_key, "")
        if not argv_class:
            # the user passed a config as a str
            class_path = passed_args[f"{argv_key}.class_path"]
            init_args_key = f"{argv_key}.init_args"
            init_args = {k[len(init_args_key) + 1 :]: v for k, v in passed_args.items() if k.startswith(init_args_key)}
            config = str({"class_path": class_path, "init_args": init_args})
        elif argv_class.startswith("{") or argv_class in ("None", "True", "False"):
            # the user passed a config as a dict
            config = argv_class
        else:
            # the user passed the shorthand format
            init_args = {k[len(argv_key) + 1 :]: v for k, v in passed_args.items()}  # +1 to account for the period
            for cls in classes:
                if cls.__name__ == argv_class:
                    config = str(_global_add_class_path(cls, init_args))
                    break
            else:
                raise ValueError(f"Could not generate a config for {repr(argv_class)}")
        return clean_argv + [argv_key, config]

    @staticmethod
    def _convert_argv_issue_85(classes: Tuple[Type, ...], nested_key: str, argv: List[str]) -> List[str]:
        """Placeholder for https://github.com/omni-us/jsonargparse/issues/85.

        Adds support for shorthand notation for ``List[object]`` arguments.
        """
        passed_args, clean_argv = [], []
        passed_configs = {}
        argv_key = f"--{nested_key}"
        # get the argv args for this nested key
        i = 0
        while i < len(argv):
            arg = argv[i]
            if arg.startswith(argv_key):
                if "=" in arg:
                    key, value = arg.split("=")
                else:
                    key = arg
                    i += 1
                    value = argv[i]
                if "class_path" in value:
                    # the user passed a config as a dict
                    passed_configs[key] = yaml.safe_load(value)
                else:
                    passed_args.append((key, value))
            else:
                clean_argv.append(arg)
            i += 1
        # generate the associated config file
        config = []
        i, n = 0, len(passed_args)
        while i < n - 1:
            ki, vi = passed_args[i]
            # convert class name to class path
            for cls in classes:
                if cls.__name__ == vi:
                    cls_type = cls
                    break
            else:
                raise ValueError(f"Could not generate a config for {repr(vi)}")
            config.append(_global_add_class_path(cls_type))
            # get any init args
            j = i + 1  # in case the j-loop doesn't run
            for j in range(i + 1, n):
                kj, vj = passed_args[j]
                if ki == kj:
                    break
                if kj.startswith(ki):
                    init_arg_name = kj.split(".")[-1]
                    config[-1]["init_args"][init_arg_name] = vj
            i = j
        # update at the end to preserve the order
        for k, v in passed_configs.items():
            config.extend(v)
        if not config:
            return clean_argv
        return clean_argv + [argv_key, str(config)]


class SaveConfigCallback(Callback):
    """Saves a LightningCLI config to the log_dir when training starts.

    Args:
        parser: The parser object used to parse the configuration.
        config: The parsed configuration that will be saved.
        config_filename: Filename for the config file.
        overwrite: Whether to overwrite an existing config file.
        multifile: When input is multiple config files, saved config preserves this structure.

    Raises:
        RuntimeError: If the config file already exists in the directory to avoid overwriting a previous run
    """

    def __init__(
        self,
        parser: LightningArgumentParser,
        config: Namespace,
        config_filename: str,
        overwrite: bool = False,
        multifile: bool = False,
    ) -> None:
        self.parser = parser
        self.config = config
        self.config_filename = config_filename
        self.overwrite = overwrite
        self.multifile = multifile

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None) -> None:
        log_dir = trainer.log_dir  # this broadcasts the directory
        assert log_dir is not None
        config_path = os.path.join(log_dir, self.config_filename)
        fs = get_filesystem(log_dir)

        if not self.overwrite:
            # check if the file exists on rank 0
            file_exists = fs.isfile(config_path) if trainer.is_global_zero else False
            # broadcast whether to fail to all ranks
            file_exists = trainer.strategy.broadcast(file_exists)
            if file_exists:
                raise RuntimeError(
                    f"{self.__class__.__name__} expected {config_path} to NOT exist. Aborting to avoid overwriting"
                    " results of a previous run. You can delete the previous config file,"
                    " set `LightningCLI(save_config_callback=None)` to disable config saving,"
                    " or set `LightningCLI(save_config_overwrite=True)` to overwrite the config file."
                )

        # save the file on rank 0
        if trainer.is_global_zero:
            # save only on rank zero to avoid race conditions.
            # the `log_dir` needs to be created as we rely on the logger to do it usually
            # but it hasn't logged anything at this point
            fs.makedirs(log_dir, exist_ok=True)
            self.parser.save(
                self.config, config_path, skip_none=False, overwrite=self.overwrite, multifile=self.multifile
            )


class LightningCLI:
    """Implementation of a configurable command line tool for pytorch-lightning."""

    def __init__(
        self,
        model_class: Optional[Union[Type[LightningModule], Callable[..., LightningModule]]] = None,
        datamodule_class: Optional[Union[Type[LightningDataModule], Callable[..., LightningDataModule]]] = None,
        save_config_callback: Optional[Type[SaveConfigCallback]] = SaveConfigCallback,
        save_config_filename: str = "config.yaml",
        save_config_overwrite: bool = False,
        save_config_multifile: bool = False,
        trainer_class: Union[Type[Trainer], Callable[..., Trainer]] = Trainer,
        trainer_defaults: Optional[Dict[str, Any]] = None,
        seed_everything_default: Optional[int] = None,
        description: str = "pytorch-lightning trainer command line tool",
        env_prefix: str = "PL",
        env_parse: bool = False,
        parser_kwargs: Optional[Union[Dict[str, Any], Dict[str, Dict[str, Any]]]] = None,
        subclass_mode_model: bool = False,
        subclass_mode_data: bool = False,
        run: bool = True,
        auto_registry: bool = False,
    ) -> None:
        """Receives as input pytorch-lightning classes (or callables which return pytorch-lightning classes), which
        are called / instantiated using a parsed configuration file and / or command line args.

        Parsing of configuration from environment variables can be enabled by setting ``env_parse=True``.
        A full configuration yaml would be parsed from ``PL_CONFIG`` if set.
        Individual settings are so parsed from variables named for example ``PL_TRAINER__MAX_EPOCHS``.

        For more info, read :ref:`the CLI docs <common/lightning_cli:LightningCLI>`.

        .. warning:: ``LightningCLI`` is in beta and subject to change.

        Args:
            model_class: An optional :class:`~pytorch_lightning.core.lightning.LightningModule` class to train on or a
                callable which returns a :class:`~pytorch_lightning.core.lightning.LightningModule` instance when
                called. If ``None``, you can pass a registered model with ``--model=MyModel``.
            datamodule_class: An optional :class:`~pytorch_lightning.core.datamodule.LightningDataModule` class or a
                callable which returns a :class:`~pytorch_lightning.core.datamodule.LightningDataModule` instance when
                called. If ``None``, you can pass a registered datamodule with ``--data=MyDataModule``.
            save_config_callback: A callback class to save the training config.
            save_config_filename: Filename for the config file.
            save_config_overwrite: Whether to overwrite an existing config file.
            save_config_multifile: When input is multiple config files, saved config preserves this structure.
            trainer_class: An optional subclass of the :class:`~pytorch_lightning.trainer.trainer.Trainer` class or a
                callable which returns a :class:`~pytorch_lightning.trainer.trainer.Trainer` instance when called.
            trainer_defaults: Set to override Trainer defaults or add persistent callbacks. The callbacks added through
                this argument will not be configurable from a configuration file and will always be present for
                this particular CLI. Alternatively, configurable callbacks can be added as explained in
                :ref:`the CLI docs <common/lightning_cli:Configurable callbacks>`.
            seed_everything_default: Default value for the :func:`~pytorch_lightning.utilities.seed.seed_everything`
                seed argument.
            description: Description of the tool shown when running ``--help``.
            env_prefix: Prefix for environment variables.
            env_parse: Whether environment variable parsing is enabled.
            parser_kwargs: Additional arguments to instantiate each ``LightningArgumentParser``.
            subclass_mode_model: Whether model can be any `subclass
                <https://jsonargparse.readthedocs.io/en/stable/#class-type-and-sub-classes>`_
                of the given class.
            subclass_mode_data: Whether datamodule can be any `subclass
                <https://jsonargparse.readthedocs.io/en/stable/#class-type-and-sub-classes>`_
                of the given class.
            run: Whether subcommands should be added to run a :class:`~pytorch_lightning.trainer.trainer.Trainer`
                method. If set to ``False``, the trainer and model classes will be instantiated only.
            auto_registry: Whether to automatically fill up the registries with all defined subclasses.
        """
        self.save_config_callback = save_config_callback
        self.save_config_filename = save_config_filename
        self.save_config_overwrite = save_config_overwrite
        self.save_config_multifile = save_config_multifile
        self.trainer_class = trainer_class
        self.trainer_defaults = trainer_defaults or {}
        self.seed_everything_default = seed_everything_default

        self.model_class = model_class
        # used to differentiate between the original value and the processed value
        self._model_class = model_class or LightningModule
        self.subclass_mode_model = (model_class is None) or subclass_mode_model

        self.datamodule_class = datamodule_class
        # used to differentiate between the original value and the processed value
        self._datamodule_class = datamodule_class or LightningDataModule
        self.subclass_mode_data = (datamodule_class is None) or subclass_mode_data

        _populate_registries(auto_registry)

        main_kwargs, subparser_kwargs = self._setup_parser_kwargs(
            parser_kwargs or {},  # type: ignore  # github.com/python/mypy/issues/6463
            {"description": description, "env_prefix": env_prefix, "default_env": env_parse},
        )
        self.setup_parser(run, main_kwargs, subparser_kwargs)
        self.parse_arguments(self.parser)

        self.subcommand = self.config["subcommand"] if run else None

        seed = self._get(self.config, "seed_everything")
        if seed is not None:
            seed_everything(seed, workers=True)

        self.before_instantiate_classes()
        self.instantiate_classes()

        if self.subcommand is not None:
            self._run_subcommand(self.subcommand)

    def _setup_parser_kwargs(
        self, kwargs: Dict[str, Any], defaults: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if kwargs.keys() & self.subcommands().keys():
            # `kwargs` contains arguments per subcommand
            return defaults, kwargs
        main_kwargs = defaults
        main_kwargs.update(kwargs)
        return main_kwargs, {}

    def init_parser(self, **kwargs: Any) -> LightningArgumentParser:
        """Method that instantiates the argument parser."""
        return LightningArgumentParser(**kwargs)

    def setup_parser(
        self, add_subcommands: bool, main_kwargs: Dict[str, Any], subparser_kwargs: Dict[str, Any]
    ) -> None:
        """Initialize and setup the parser, subcommands, and arguments."""
        self.parser = self.init_parser(**main_kwargs)
        if add_subcommands:
            self._subcommand_method_arguments: Dict[str, List[str]] = {}
            self._add_subcommands(self.parser, **subparser_kwargs)
        else:
            self._add_arguments(self.parser)

    def add_default_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Adds default arguments to the parser."""
        parser.add_argument(
            "--seed_everything",
            type=Optional[int],
            default=self.seed_everything_default,
            help="Set to an int to run seed_everything with this value before classes instantiation",
        )

    def add_core_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Adds arguments from the core classes to the parser."""
        parser.add_lightning_class_args(self.trainer_class, "trainer")
        parser.set_choices("trainer.callbacks", CALLBACK_REGISTRY.classes, is_list=True)
        parser.set_choices("trainer.logger", LOGGER_REGISTRY.classes)
        trainer_defaults = {"trainer." + k: v for k, v in self.trainer_defaults.items() if k != "callbacks"}
        parser.set_defaults(trainer_defaults)

        parser.add_lightning_class_args(self._model_class, "model", subclass_mode=self.subclass_mode_model)
        if self.model_class is None and len(MODEL_REGISTRY):
            # did not pass a model and there are models registered
            parser.set_choices("model", MODEL_REGISTRY.classes)

        if self.datamodule_class is not None:
            parser.add_lightning_class_args(self._datamodule_class, "data", subclass_mode=self.subclass_mode_data)
        elif len(DATAMODULE_REGISTRY):
            # this should not be required because the user might want to use the `LightningModule` dataloaders
            parser.add_lightning_class_args(
                self._datamodule_class, "data", subclass_mode=self.subclass_mode_data, required=False
            )
            parser.set_choices("data", DATAMODULE_REGISTRY.classes)

    def _add_arguments(self, parser: LightningArgumentParser) -> None:
        # default + core + custom arguments
        self.add_default_arguments_to_parser(parser)
        self.add_core_arguments_to_parser(parser)
        self.add_arguments_to_parser(parser)
        # add default optimizer args if necessary
        if not parser._optimizers:  # already added by the user in `add_arguments_to_parser`
            parser.add_optimizer_args(OPTIMIZER_REGISTRY.classes)
        if not parser._lr_schedulers:  # already added by the user in `add_arguments_to_parser`
            parser.add_lr_scheduler_args(LR_SCHEDULER_REGISTRY.classes)
        self.link_optimizers_and_lr_schedulers(parser)

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Implement to add extra arguments to the parser or link arguments.

        Args:
            parser: The parser object to which arguments can be added
        """

    @staticmethod
    def subcommands() -> Dict[str, Set[str]]:
        """Defines the list of available subcommands and the arguments to skip."""
        return {
            "fit": {"model", "train_dataloaders", "val_dataloaders", "datamodule"},
            "validate": {"model", "dataloaders", "datamodule"},
            "test": {"model", "dataloaders", "datamodule"},
            "predict": {"model", "dataloaders", "datamodule"},
            "tune": {"model", "train_dataloaders", "val_dataloaders", "datamodule"},
        }

    def _add_subcommands(self, parser: LightningArgumentParser, **kwargs: Any) -> None:
        """Adds subcommands to the input parser."""
        parser_subcommands = parser.add_subcommands()
        # the user might have passed a builder function
        trainer_class = (
            self.trainer_class if isinstance(self.trainer_class, type) else class_from_function(self.trainer_class)
        )
        # register all subcommands in separate subcommand parsers under the main parser
        for subcommand in self.subcommands():
            subcommand_parser = self._prepare_subcommand_parser(trainer_class, subcommand, **kwargs.get(subcommand, {}))
            fn = getattr(trainer_class, subcommand)
            # extract the first line description in the docstring for the subcommand help message
            description = _get_short_description(fn)
            parser_subcommands.add_subcommand(subcommand, subcommand_parser, help=description)

    def _prepare_subcommand_parser(self, klass: Type, subcommand: str, **kwargs: Any) -> LightningArgumentParser:
        parser = self.init_parser(**kwargs)
        self._add_arguments(parser)
        # subcommand arguments
        skip = self.subcommands()[subcommand]
        added = parser.add_method_arguments(klass, subcommand, skip=skip)
        # need to save which arguments were added to pass them to the method later
        self._subcommand_method_arguments[subcommand] = added
        return parser

    @staticmethod
    def link_optimizers_and_lr_schedulers(parser: LightningArgumentParser) -> None:
        """Creates argument links for optimizers and learning rate schedulers that specified a ``link_to``."""
        optimizers_and_lr_schedulers = {**parser._optimizers, **parser._lr_schedulers}
        for key, (class_type, link_to) in optimizers_and_lr_schedulers.items():
            if link_to == "AUTOMATIC":
                continue
            if isinstance(class_type, tuple):
                parser.link_arguments(key, link_to)
            else:
                add_class_path = _add_class_path_generator(class_type)
                parser.link_arguments(key, link_to, compute_fn=add_class_path)

    def parse_arguments(self, parser: LightningArgumentParser) -> None:
        """Parses command line arguments and stores it in ``self.config``."""
        self.config = parser.parse_args()

    def before_instantiate_classes(self) -> None:
        """Implement to run some code before instantiating the classes."""

    def instantiate_classes(self) -> None:
        """Instantiates the classes and sets their attributes."""
        self.config_init = self.parser.instantiate_classes(self.config)
        self.datamodule = self._get(self.config_init, "data")
        self.model = self._get(self.config_init, "model")
        self._add_configure_optimizers_method_to_model(self.subcommand)
        self.trainer = self.instantiate_trainer()

    def instantiate_trainer(self, **kwargs: Any) -> Trainer:
        """Instantiates the trainer.

        Args:
            kwargs: Any custom trainer arguments.
        """
        extra_callbacks = [self._get(self.config_init, c) for c in self._parser(self.subcommand).callback_keys]
        trainer_config = {**self._get(self.config_init, "trainer"), **kwargs}
        return self._instantiate_trainer(trainer_config, extra_callbacks)

    def _instantiate_trainer(self, config: Dict[str, Any], callbacks: List[Callback]) -> Trainer:
        config["callbacks"] = config["callbacks"] or []
        config["callbacks"].extend(callbacks)
        if "callbacks" in self.trainer_defaults:
            if isinstance(self.trainer_defaults["callbacks"], list):
                config["callbacks"].extend(self.trainer_defaults["callbacks"])
            else:
                config["callbacks"].append(self.trainer_defaults["callbacks"])
        if self.save_config_callback and not config["fast_dev_run"]:
            config_callback = self.save_config_callback(
                self._parser(self.subcommand),
                self.config.get(str(self.subcommand), self.config),
                self.save_config_filename,
                overwrite=self.save_config_overwrite,
                multifile=self.save_config_multifile,
            )
            config["callbacks"].append(config_callback)
        return self.trainer_class(**config)

    def _parser(self, subcommand: Optional[str]) -> LightningArgumentParser:
        if subcommand is None:
            return self.parser
        # return the subcommand parser for the subcommand passed
        action_subcommand = self.parser._subcommands_action
        return action_subcommand._name_parser_map[subcommand]

    @staticmethod
    def configure_optimizers(
        lightning_module: LightningModule, optimizer: Optimizer, lr_scheduler: Optional[LRSchedulerTypeUnion] = None
    ) -> Any:
        """Override to customize the :meth:`~pytorch_lightning.core.lightning.LightningModule.configure_optimizers`
        method.

        Args:
            lightning_module: A reference to the model.
            optimizer: The optimizer.
            lr_scheduler: The learning rate scheduler (if used).
        """
        if lr_scheduler is None:
            return optimizer
        if isinstance(lr_scheduler, ReduceLROnPlateau):
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, "monitor": lr_scheduler.monitor},
            }
        return [optimizer], [lr_scheduler]

    def _add_configure_optimizers_method_to_model(self, subcommand: Optional[str]) -> None:
        """Overrides the model's :meth:`~pytorch_lightning.core.lightning.LightningModule.configure_optimizers`
        method if a single optimizer and optionally a scheduler argument groups are added to the parser as
        'AUTOMATIC'."""
        parser = self._parser(subcommand)

        def get_automatic(
            class_type: Union[Type, Tuple[Type, ...]], register: Dict[str, Tuple[Union[Type, Tuple[Type, ...]], str]]
        ) -> List[str]:
            automatic = []
            for key, (base_class, link_to) in register.items():
                if not isinstance(base_class, tuple):
                    base_class = (base_class,)
                if link_to == "AUTOMATIC" and any(issubclass(c, class_type) for c in base_class):
                    automatic.append(key)
            return automatic

        optimizers = get_automatic(Optimizer, parser._optimizers)
        lr_schedulers = get_automatic(LRSchedulerTypeTuple, parser._lr_schedulers)

        if len(optimizers) == 0:
            return

        if len(optimizers) > 1 or len(lr_schedulers) > 1:
            raise MisconfigurationException(
                f"`{self.__class__.__name__}.add_configure_optimizers_method_to_model` expects at most one optimizer "
                f"and one lr_scheduler to be 'AUTOMATIC', but found {optimizers+lr_schedulers}. In this case the user "
                "is expected to link the argument groups and implement `configure_optimizers`, see "
                "https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_cli.html"
                "#optimizers-and-learning-rate-schedulers"
            )

        optimizer_class = parser._optimizers[optimizers[0]][0]
        optimizer_init = self._get(self.config_init, optimizers[0])
        if not isinstance(optimizer_class, tuple):
            optimizer_init = _global_add_class_path(optimizer_class, optimizer_init)
        if not optimizer_init:
            # optimizers were registered automatically but not passed by the user
            return

        lr_scheduler_init = None
        if lr_schedulers:
            lr_scheduler_class = parser._lr_schedulers[lr_schedulers[0]][0]
            lr_scheduler_init = self._get(self.config_init, lr_schedulers[0])
            if not isinstance(lr_scheduler_class, tuple):
                lr_scheduler_init = _global_add_class_path(lr_scheduler_class, lr_scheduler_init)

        if is_overridden("configure_optimizers", self.model):
            _warn(
                f"`{self.model.__class__.__name__}.configure_optimizers` will be overridden by "
                f"`{self.__class__.__name__}.configure_optimizers`."
            )

        optimizer = instantiate_class(self.model.parameters(), optimizer_init)
        lr_scheduler = instantiate_class(optimizer, lr_scheduler_init) if lr_scheduler_init else None
        fn = partial(self.configure_optimizers, optimizer=optimizer, lr_scheduler=lr_scheduler)
        update_wrapper(fn, self.configure_optimizers)  # necessary for `is_overridden`
        # override the existing method
        self.model.configure_optimizers = MethodType(fn, self.model)

    def _get(self, config: Dict[str, Any], key: str, default: Optional[Any] = None) -> Any:
        """Utility to get a config value which might be inside a subcommand."""
        return config.get(str(self.subcommand), config).get(key, default)

    def _run_subcommand(self, subcommand: str) -> None:
        """Run the chosen subcommand."""
        before_fn = getattr(self, f"before_{subcommand}", None)
        if callable(before_fn):
            before_fn()

        default = getattr(self.trainer, subcommand)
        fn = getattr(self, subcommand, default)
        fn_kwargs = self._prepare_subcommand_kwargs(subcommand)
        fn(**fn_kwargs)

        after_fn = getattr(self, f"after_{subcommand}", None)
        if callable(after_fn):
            after_fn()

    def _prepare_subcommand_kwargs(self, subcommand: str) -> Dict[str, Any]:
        """Prepares the keyword arguments to pass to the subcommand to run."""
        fn_kwargs = {
            k: v for k, v in self.config_init[subcommand].items() if k in self._subcommand_method_arguments[subcommand]
        }
        fn_kwargs["model"] = self.model
        if self.datamodule is not None:
            fn_kwargs["datamodule"] = self.datamodule
        return fn_kwargs


def _class_path_from_class(class_type: Type) -> str:
    return class_type.__module__ + "." + class_type.__name__


def _global_add_class_path(
    class_type: Type, init_args: Optional[Union[Namespace, Dict[str, Any]]] = None
) -> Dict[str, Any]:
    if isinstance(init_args, Namespace):
        init_args = init_args.as_dict()
    return {"class_path": _class_path_from_class(class_type), "init_args": init_args or {}}


def _add_class_path_generator(class_type: Type) -> Callable[[Namespace], Dict[str, Any]]:
    def add_class_path(init_args: Namespace) -> Dict[str, Any]:
        return _global_add_class_path(class_type, init_args)

    return add_class_path


def instantiate_class(args: Union[Any, Tuple[Any, ...]], init: Dict[str, Any]) -> Any:
    """Instantiates a class with the given args and init.

    Args:
        args: Positional arguments required for instantiation.
        init: Dict of the form {"class_path":...,"init_args":...}.

    Returns:
        The instantiated class object.
    """
    kwargs = init.get("init_args", {})
    if not isinstance(args, tuple):
        args = (args,)
    class_module, class_name = init["class_path"].rsplit(".", 1)
    module = __import__(class_module, fromlist=[class_name])
    args_class = getattr(module, class_name)
    return args_class(*args, **kwargs)


def _get_short_description(component: object) -> Optional[str]:
    if component.__doc__ is None:
        return None
    if not _DOCSTRING_PARSER_AVAILABLE:
        rank_zero_warn(f"Failed parsing docstring for {component}: docstring-parser package is required")
    else:
        try:
            docstring = docstring_parser.parse(component.__doc__)
            return docstring.short_description
        except (ValueError, docstring_parser.ParseError) as ex:
            rank_zero_warn(f"Failed parsing docstring for {component}: {ex}")
