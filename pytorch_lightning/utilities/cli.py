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
import inspect
import os
import warnings
from argparse import Namespace
from types import MethodType
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Type, Union

from torch.optim import Optimizer

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.utilities import _module_available
from pytorch_lightning.utilities.cloud_io import get_filesystem
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.utilities.types import LRSchedulerType, LRSchedulerTypeTuple

_JSONARGPARSE_AVAILABLE = _module_available("jsonargparse")
if _JSONARGPARSE_AVAILABLE:
    from jsonargparse import ActionConfigFile, ArgumentParser, class_from_function, set_config_read_mode

    set_config_read_mode(fsspec_enabled=True)
else:
    ArgumentParser = object


class LightningArgumentParser(ArgumentParser):
    """Extension of jsonargparse's ArgumentParser for pytorch-lightning"""

    def __init__(self, *args: Any, parse_as_dict: bool = True, **kwargs: Any) -> None:
        """Initialize argument parser that supports configuration file input

        For full details of accepted arguments see `ArgumentParser.__init__
        <https://jsonargparse.readthedocs.io/en/stable/#jsonargparse.core.ArgumentParser.__init__>`_.
        """
        if not _JSONARGPARSE_AVAILABLE:
            raise ModuleNotFoundError(
                "`jsonargparse` is not installed but it is required for the CLI."
                " Install it with `pip install jsonargparse[signatures]`."
            )
        super().__init__(*args, parse_as_dict=parse_as_dict, **kwargs)
        self.add_argument(
            "--config", action=ActionConfigFile, help="Path to a configuration file in json or yaml format."
        )
        self.callback_keys: List[str] = []
        self.optimizers_and_lr_schedulers: Dict[str, Tuple[Union[Type, Tuple[Type, ...]], str]] = {}

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
    ) -> List[str]:
        """
        Adds arguments from a lightning class to a nested key of the parser

        Args:
            lightning_class: A callable or any subclass of {Trainer, LightningModule, LightningDataModule, Callback}.
            nested_key: Name of the nested namespace to store arguments.
            subclass_mode: Whether allow any subclass of the given class.
        """
        if callable(lightning_class) and not inspect.isclass(lightning_class):
            lightning_class = class_from_function(lightning_class)

        lightning_class = cast(type, lightning_class)
        if inspect.isclass(lightning_class) and issubclass(
            lightning_class, (Trainer, LightningModule, LightningDataModule, Callback)
        ):
            if issubclass(lightning_class, Callback):
                self.callback_keys.append(nested_key)
            if subclass_mode:
                return self.add_subclass_arguments(lightning_class, nested_key, required=True)
            return self.add_class_arguments(
                lightning_class, nested_key, fail_untyped=False, instantiate=not issubclass(lightning_class, Trainer)
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
        """
        Adds arguments from an optimizer class to a nested key of the parser

        Args:
            optimizer_class: Any subclass of torch.optim.Optimizer.
            nested_key: Name of the nested namespace to store arguments.
            link_to: Dot notation of a parser key to set arguments or AUTOMATIC.
        """
        if isinstance(optimizer_class, tuple):
            assert all(issubclass(o, Optimizer) for o in optimizer_class)
        else:
            assert issubclass(optimizer_class, Optimizer)
        kwargs = {"instantiate": False, "fail_untyped": False, "skip": {"params"}}
        if isinstance(optimizer_class, tuple):
            self.add_subclass_arguments(optimizer_class, nested_key, required=True, **kwargs)
        else:
            self.add_class_arguments(optimizer_class, nested_key, **kwargs)
        self.optimizers_and_lr_schedulers[nested_key] = (optimizer_class, link_to)

    def add_lr_scheduler_args(
        self,
        lr_scheduler_class: Union[LRSchedulerType, Tuple[LRSchedulerType, ...]],
        nested_key: str = "lr_scheduler",
        link_to: str = "AUTOMATIC",
    ) -> None:
        """
        Adds arguments from a learning rate scheduler class to a nested key of the parser

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
            self.add_subclass_arguments(lr_scheduler_class, nested_key, required=True, **kwargs)
        else:
            self.add_class_arguments(lr_scheduler_class, nested_key, **kwargs)
        self.optimizers_and_lr_schedulers[nested_key] = (lr_scheduler_class, link_to)


class SaveConfigCallback(Callback):
    """Saves a LightningCLI config to the log_dir when training starts

    Raises:
        RuntimeError: If the config file already exists in the directory to avoid overwriting a previous run
    """

    def __init__(
        self,
        parser: LightningArgumentParser,
        config: Union[Namespace, Dict[str, Any]],
        config_filename: str,
        overwrite: bool = False,
    ) -> None:
        self.parser = parser
        self.config = config
        self.config_filename = config_filename
        self.overwrite = overwrite

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None) -> None:
        # save the config in `setup` because (1) we want it to save regardless of the trainer function run
        # and we want to save before processes are spawned
        log_dir = trainer.log_dir
        assert log_dir is not None
        config_path = os.path.join(log_dir, self.config_filename)
        if not self.overwrite and os.path.isfile(config_path):
            raise RuntimeError(
                f"{self.__class__.__name__} expected {config_path} to NOT exist. Aborting to avoid overwriting"
                " results of a previous run. You can delete the previous config file,"
                " set `LightningCLI(save_config_callback=None)` to disable config saving,"
                " or set `LightningCLI(save_config_overwrite=True)` to overwrite the config file."
            )
        if trainer.is_global_zero:
            # save only on rank zero to avoid race conditions on DDP.
            # the `log_dir` needs to be created as we rely on the logger to do it usually
            # but it hasn't logged anything at this point
            get_filesystem(log_dir).makedirs(log_dir, exist_ok=True)
            self.parser.save(self.config, config_path, skip_none=False, overwrite=self.overwrite)

    def __reduce__(self) -> Tuple[Type["SaveConfigCallback"], Tuple, Dict]:
        # `ArgumentParser` is un-pickleable. Drop it
        return (self.__class__, (None, self.config, self.config_filename), {})


class LightningCLI:
    """Implementation of a configurable command line tool for pytorch-lightning"""

    def __init__(
        self,
        model_class: Union[Type[LightningModule], Callable[..., LightningModule]],
        datamodule_class: Optional[Union[Type[LightningDataModule], Callable[..., LightningDataModule]]] = None,
        save_config_callback: Optional[Type[SaveConfigCallback]] = SaveConfigCallback,
        save_config_filename: str = "config.yaml",
        save_config_overwrite: bool = False,
        trainer_class: Union[Type[Trainer], Callable[..., Trainer]] = Trainer,
        trainer_defaults: Dict[str, Any] = None,
        seed_everything_default: int = None,
        description: str = "pytorch-lightning trainer command line tool",
        env_prefix: str = "PL",
        env_parse: bool = False,
        parser_kwargs: Dict[str, Any] = None,
        subclass_mode_model: bool = False,
        subclass_mode_data: bool = False,
    ) -> None:
        """
        Receives as input pytorch-lightning classes (or callables which return pytorch-lightning classes), which are
        called / instantiated using a parsed configuration file and / or command line args and then runs trainer.fit.
        Parsing of configuration from environment variables can be enabled by setting ``env_parse=True``. A full
        configuration yaml would be parsed from ``PL_CONFIG`` if set. Individual settings are so parsed from variables
        named for example ``PL_TRAINER__MAX_EPOCHS``.

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
            model_class: :class:`~pytorch_lightning.core.lightning.LightningModule` class to train on or a callable
                which returns a :class:`~pytorch_lightning.core.lightning.LightningModule` instance when called.
            datamodule_class: An optional :class:`~pytorch_lightning.core.datamodule.LightningDataModule` class or a
                callable which returns a :class:`~pytorch_lightning.core.datamodule.LightningDataModule` instance when
                called.
            save_config_callback: A callback class to save the training config.
            save_config_filename: Filename for the config file.
            save_config_overwrite: Whether to overwrite an existing config file.
            trainer_class: An optional subclass of the :class:`~pytorch_lightning.trainer.trainer.Trainer` class or a
                callable which returns a :class:`~pytorch_lightning.trainer.trainer.Trainer` instance when called.
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
        self.model_class = model_class
        self.datamodule_class = datamodule_class
        self.save_config_callback = save_config_callback
        self.save_config_filename = save_config_filename
        self.save_config_overwrite = save_config_overwrite
        self.trainer_class = trainer_class
        self.trainer_defaults = {} if trainer_defaults is None else trainer_defaults
        self.seed_everything_default = seed_everything_default
        self.subclass_mode_model = subclass_mode_model
        self.subclass_mode_data = subclass_mode_data
        self.parser_kwargs = {} if parser_kwargs is None else parser_kwargs
        self.parser_kwargs.update({"description": description, "env_prefix": env_prefix, "default_env": env_parse})

        self.init_parser()
        self.add_core_arguments_to_parser()
        self.add_arguments_to_parser(self.parser)
        self.link_optimizers_and_lr_schedulers()
        self.parse_arguments()
        if self.config["seed_everything"] is not None:
            seed_everything(self.config["seed_everything"], workers=True)
        self.before_instantiate_classes()
        self.instantiate_classes()
        self.add_configure_optimizers_method_to_model()
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
            "--seed_everything",
            type=Optional[int],
            default=self.seed_everything_default,
            help="Set to an int to run seed_everything with this value before classes instantiation",
        )
        self.parser.add_lightning_class_args(self.trainer_class, "trainer")
        trainer_defaults = {"trainer." + k: v for k, v in self.trainer_defaults.items() if k != "callbacks"}
        self.parser.set_defaults(trainer_defaults)
        self.parser.add_lightning_class_args(self.model_class, "model", subclass_mode=self.subclass_mode_model)
        if self.datamodule_class is not None:
            self.parser.add_lightning_class_args(self.datamodule_class, "data", subclass_mode=self.subclass_mode_data)

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Implement to add extra arguments to parser or link arguments

        Args:
            parser: The argument parser object to which arguments can be added
        """

    def link_optimizers_and_lr_schedulers(self) -> None:
        """Creates argument links for optimizers and lr_schedulers that specified a link_to"""
        for key, (class_type, link_to) in self.parser.optimizers_and_lr_schedulers.items():
            if link_to == "AUTOMATIC":
                continue
            if isinstance(class_type, tuple):
                self.parser.link_arguments(key, link_to)
            else:
                add_class_path = _add_class_path_generator(class_type)
                self.parser.link_arguments(key, link_to, compute_fn=add_class_path)

    def parse_arguments(self) -> None:
        """Parses command line arguments and stores it in self.config"""
        self.config = self.parser.parse_args()

    def before_instantiate_classes(self) -> None:
        """Implement to run some code before instantiating the classes"""

    def instantiate_classes(self) -> None:
        """Instantiates the classes using settings from self.config"""
        self.config_init = self.parser.instantiate_classes(self.config)
        self.datamodule = self.config_init.get("data")
        self.model = self.config_init["model"]
        self.instantiate_trainer()

    def instantiate_trainer(self) -> None:
        """Instantiates the trainer using self.config_init['trainer']"""
        if self.config_init["trainer"].get("callbacks") is None:
            self.config_init["trainer"]["callbacks"] = []
        callbacks = [self.config_init[c] for c in self.parser.callback_keys]
        self.config_init["trainer"]["callbacks"].extend(callbacks)
        if "callbacks" in self.trainer_defaults:
            if isinstance(self.trainer_defaults["callbacks"], list):
                self.config_init["trainer"]["callbacks"].extend(self.trainer_defaults["callbacks"])
            else:
                self.config_init["trainer"]["callbacks"].append(self.trainer_defaults["callbacks"])
        if self.save_config_callback and not self.config_init["trainer"]["fast_dev_run"]:
            config_callback = self.save_config_callback(
                self.parser, self.config, self.save_config_filename, overwrite=self.save_config_overwrite
            )
            self.config_init["trainer"]["callbacks"].append(config_callback)
        self.trainer = self.trainer_class(**self.config_init["trainer"])

    def add_configure_optimizers_method_to_model(self) -> None:
        """
        Adds to the model an automatically generated configure_optimizers method

        If a single optimizer and optionally a scheduler argument groups are added to the parser as 'AUTOMATIC',
        then a `configure_optimizers` method is automatically implemented in the model class.
        """

        def get_automatic(class_type: Union[Type, Tuple[Type, ...]]) -> List[str]:
            automatic = []
            for key, (base_class, link_to) in self.parser.optimizers_and_lr_schedulers.items():
                if not isinstance(base_class, tuple):
                    base_class = (base_class,)
                if link_to == "AUTOMATIC" and any(issubclass(c, class_type) for c in base_class):
                    automatic.append(key)
            return automatic

        optimizers = get_automatic(Optimizer)
        lr_schedulers = get_automatic(LRSchedulerTypeTuple)

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

        if is_overridden("configure_optimizers", self.model):
            warnings.warn(
                f"`{self.model.__class__.__name__}.configure_optimizers` will be overridden by "
                f"`{self.__class__.__name__}.add_configure_optimizers_method_to_model`."
            )

        optimizer_class = self.parser.optimizers_and_lr_schedulers[optimizers[0]][0]
        optimizer_init = self.config_init.get(optimizers[0], {})
        if not isinstance(optimizer_class, tuple):
            optimizer_init = _global_add_class_path(optimizer_class, optimizer_init)
        lr_scheduler_init = None
        if lr_schedulers:
            lr_scheduler_class = self.parser.optimizers_and_lr_schedulers[lr_schedulers[0]][0]
            lr_scheduler_init = self.config_init.get(lr_schedulers[0], {})
            if not isinstance(lr_scheduler_class, tuple):
                lr_scheduler_init = _global_add_class_path(lr_scheduler_class, lr_scheduler_init)

        def configure_optimizers(
            self: LightningModule,
        ) -> Union[Optimizer, Tuple[List[Optimizer], List[LRSchedulerType]]]:
            optimizer = instantiate_class(self.parameters(), optimizer_init)
            if not lr_scheduler_init:
                return optimizer
            lr_scheduler = instantiate_class(optimizer, lr_scheduler_init)
            return [optimizer], [lr_scheduler]

        self.model.configure_optimizers = MethodType(configure_optimizers, self.model)

    def prepare_fit_kwargs(self) -> None:
        """Prepares fit_kwargs including datamodule using self.config_init['data'] if given"""
        self.fit_kwargs = {"model": self.model}
        if self.datamodule is not None:
            self.fit_kwargs["datamodule"] = self.datamodule

    def before_fit(self) -> None:
        """Implement to run some code before fit is started"""

    def fit(self) -> None:
        """Runs fit of the instantiated trainer class and prepared fit keyword arguments"""
        self.trainer.fit(**self.fit_kwargs)

    def after_fit(self) -> None:
        """Implement to run some code after fit has finished"""


def _global_add_class_path(class_type: Type, init_args: Dict[str, Any]) -> Dict[str, Any]:
    return {"class_path": class_type.__module__ + "." + class_type.__name__, "init_args": init_args}


def _add_class_path_generator(class_type: Type) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    def add_class_path(init_args: Dict[str, Any]) -> Dict[str, Any]:
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
