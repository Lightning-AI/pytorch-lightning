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
"""LightningDataModule for loading DataLoaders with ease."""

import functools
import inspect
from abc import abstractmethod
from argparse import ArgumentParser, Namespace
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

from torch.utils.data import DataLoader, Dataset

from pytorch_lightning.core.hooks import CheckpointHooks, DataHooks
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.parsing import str_to_bool, str_to_bool_or_str


class _DataModuleWrapper(type):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__has_added_checks = False

    def __call__(cls, *args, **kwargs):
        """A wrapper for LightningDataModule that:

        1. Runs user defined subclass's __init__
        2. Assures prepare_data() runs on rank 0
        3. Lets you check prepare_data and setup to see if they've been called
        """
        if not cls.__has_added_checks:
            cls.__has_added_checks = True
            # Track prepare_data calls and make sure it runs on rank zero
            cls.prepare_data = track_data_hook_calls(rank_zero_only(cls.prepare_data))
            # Track setup calls
            cls.setup = track_data_hook_calls(cls.setup)
            # Track teardown calls
            cls.teardown = track_data_hook_calls(cls.teardown)

        # Get instance of LightningDataModule by mocking its __init__ via __call__
        obj = type.__call__(cls, *args, **kwargs)

        return obj


def track_data_hook_calls(fn):
    """A decorator that checks if prepare_data/setup/teardown has been called.

    - When ``dm.prepare_data()`` is called, ``dm.has_prepared_data`` gets set to True
    - When ``dm.setup()``, ``dm.has_setup_{fit,validate,test}`` get set to True
    - When ``dm.setup(stage)`` is called, where stage is any of ``{fit,validate,test,predict}``.
        Its corresponding `dm_has_setup_{stage}` attribute gets set to True
    - ``dm.teardown()`` and ``dm.teardown(stage)`` acts exactly like ``dm.setup``

    Args:
        fn (function): Function that will be tracked to see if it has been called.

    Returns:
        function: Decorated function that tracks its call status and saves it to private attrs in its obj instance.
    """

    @functools.wraps(fn)
    def wrapped_fn(*args, **kwargs):

        # The object instance from which setup or prepare_data was called
        obj = args[0]
        name = fn.__name__

        # If calling setup, we check the stage and assign stage-specific bool args
        if name in ("setup", "teardown"):

            # Get stage either by grabbing from args or checking kwargs.
            # If not provided, set call status of 'fit', 'validate', and 'test' to True.
            # We do this so __attach_datamodule in trainer.py doesn't mistakenly call setup('test') on trainer.test()
            stage = args[1] if len(args) > 1 else kwargs.get("stage", None)

            if stage is None:
                for s in ("fit", "validate", "test"):
                    setattr(obj, f"_has_{name}_{s}", True)
            else:
                setattr(obj, f"_has_{name}_{stage}", True)

        elif name == "prepare_data":
            obj._has_prepared_data = True

        return fn(*args, **kwargs)

    return wrapped_fn


class LightningDataModule(CheckpointHooks, DataHooks, metaclass=_DataModuleWrapper):
    """
    A DataModule standardizes the training, val, test splits, data preparation and transforms.
    The main advantage is consistent data splits, data preparation and transforms across models.

    Example::

        class MyDataModule(LightningDataModule):
            def __init__(self):
                super().__init__()
            def prepare_data(self):
                # download, split, etc...
                # only called on 1 GPU/TPU in distributed
            def setup(self):
                # make assignments here (val/train/test split)
                # called on every process in DDP
            def train_dataloader(self):
                train_split = Dataset(...)
                return DataLoader(train_split)
            def val_dataloader(self):
                val_split = Dataset(...)
                return DataLoader(val_split)
            def test_dataloader(self):
                test_split = Dataset(...)
                return DataLoader(test_split)
            def teardown(self):
                # clean up after fit or test
                # called on every process in DDP

    A DataModule implements 6 key methods:

    * **prepare_data** (things to do on 1 GPU/TPU not on every GPU/TPU in distributed mode).
    * **setup**  (things to do on every accelerator in distributed mode).
    * **train_dataloader** the training dataloader.
    * **val_dataloader** the val dataloader(s).
    * **test_dataloader** the test dataloader(s).
    * **teardown** (things to do on every accelerator in distributed mode when finished)


    This allows you to share a full dataset without explaining how to download,
    split transform and process the data

    """

    name: str = ...

    def __init__(
        self,
        train_transforms=None,
        val_transforms=None,
        test_transforms=None,
        dims=None,
    ):
        super().__init__()
        self._train_transforms = train_transforms
        self._val_transforms = val_transforms
        self._test_transforms = test_transforms
        self._dims = dims if dims is not None else ()

        # Pointer to the trainer object
        self.trainer = None

        # Private attrs to keep track of whether or not data hooks have been called yet
        self._has_prepared_data = False

        self._has_setup_fit = False
        self._has_setup_validate = False
        self._has_setup_test = False
        self._has_setup_predict = False

        self._has_teardown_fit = False
        self._has_teardown_validate = False
        self._has_teardown_test = False
        self._has_teardown_predict = False

    @property
    def train_transforms(self):
        """
        Optional transforms (or collection of transforms) you can apply to train dataset
        """
        return self._train_transforms

    @train_transforms.setter
    def train_transforms(self, t):
        self._train_transforms = t

    @property
    def val_transforms(self):
        """
        Optional transforms (or collection of transforms) you can apply to validation dataset
        """
        return self._val_transforms

    @val_transforms.setter
    def val_transforms(self, t):
        self._val_transforms = t

    @property
    def test_transforms(self):
        """
        Optional transforms (or collection of transforms) you can apply to test dataset
        """
        return self._test_transforms

    @test_transforms.setter
    def test_transforms(self, t):
        self._test_transforms = t

    @property
    def dims(self):
        """
        A tuple describing the shape of your data. Extra functionality exposed in ``size``.
        """
        return self._dims

    @dims.setter
    def dims(self, d):
        self._dims = d

    def size(self, dim=None) -> Union[Tuple, int]:
        """
        Return the dimension of each input either as a tuple or list of tuples. You can index this
        just as you would with a torch tensor.
        """

        if dim is not None:
            return self.dims[dim]

        return self.dims

    @property
    def has_prepared_data(self) -> bool:
        """Return bool letting you know if ``datamodule.prepare_data()`` has been called or not.

        Returns:
            bool: True if ``datamodule.prepare_data()`` has been called. False by default.
        """
        return self._has_prepared_data

    @property
    def has_setup_fit(self) -> bool:
        """Return bool letting you know if ``datamodule.setup(stage='fit')`` has been called or not.

        Returns:
            bool: True ``if datamodule.setup(stage='fit')`` has been called. False by default.
        """
        return self._has_setup_fit

    @property
    def has_setup_validate(self) -> bool:
        """Return bool letting you know if ``datamodule.setup(stage='validate')`` has been called or not.

        Returns:
            bool: True if ``datamodule.setup(stage='validate')`` has been called. False by default.
        """
        return self._has_setup_validate

    @property
    def has_setup_test(self) -> bool:
        """Return bool letting you know if ``datamodule.setup(stage='test')`` has been called or not.

        Returns:
            bool: True if ``datamodule.setup(stage='test')`` has been called. False by default.
        """
        return self._has_setup_test

    @property
    def has_setup_predict(self) -> bool:
        """Return bool letting you know if ``datamodule.setup(stage='predict')`` has been called or not.

        Returns:
            bool: True if ``datamodule.setup(stage='predict')`` has been called. False by default.
        """
        return self._has_setup_predict

    @property
    def has_teardown_fit(self) -> bool:
        """Return bool letting you know if ``datamodule.teardown(stage='fit')`` has been called or not.

        Returns:
            bool: True ``if datamodule.teardown(stage='fit')`` has been called. False by default.
        """
        return self._has_teardown_fit

    @property
    def has_teardown_validate(self) -> bool:
        """Return bool letting you know if ``datamodule.teardown(stage='validate')`` has been called or not.

        Returns:
            bool: True if ``datamodule.teardown(stage='validate')`` has been called. False by default.
        """
        return self._has_teardown_validate

    @property
    def has_teardown_test(self) -> bool:
        """Return bool letting you know if ``datamodule.teardown(stage='test')`` has been called or not.

        Returns:
            bool: True if ``datamodule.teardown(stage='test')`` has been called. False by default.
        """
        return self._has_teardown_test

    @property
    def has_teardown_predict(self) -> bool:
        """Return bool letting you know if ``datamodule.teardown(stage='predict')`` has been called or not.

        Returns:
            bool: True if ``datamodule.teardown(stage='predict')`` has been called. False by default.
        """
        return self._has_teardown_predict

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser) -> ArgumentParser:
        r"""Extends existing argparse by default `LightningDataModule` attributes."""
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        added_args = [x.dest for x in parser._actions]

        blacklist = ["kwargs"]
        depr_arg_names = blacklist + added_args
        depr_arg_names = set(depr_arg_names)

        allowed_types = (str, int, float, bool)

        # TODO: get "help" from docstring :)
        for arg, arg_types, arg_default in (
            at for at in cls.get_init_arguments_and_types() if at[0] not in depr_arg_names
        ):
            arg_types = [at for at in allowed_types if at in arg_types]
            if not arg_types:
                # skip argument with not supported type
                continue
            arg_kwargs = {}
            if bool in arg_types:
                arg_kwargs.update(nargs="?", const=True)
                # if the only arg type is bool
                if len(arg_types) == 1:
                    use_type = str_to_bool
                # if only two args (str, bool)
                elif len(arg_types) == 2 and set(arg_types) == {str, bool}:
                    use_type = str_to_bool_or_str
                else:
                    # filter out the bool as we need to use more general
                    use_type = [at for at in arg_types if at is not bool][0]
            else:
                use_type = arg_types[0]

            if arg_default == inspect._empty:
                arg_default = None

            parser.add_argument(
                f"--{arg}",
                dest=arg,
                default=arg_default,
                type=use_type,
                help=f"autogenerated by plb.{cls.__name__}",
                **arg_kwargs,
            )

        return parser

    @classmethod
    def from_argparse_args(cls, args: Union[Namespace, ArgumentParser], **kwargs):
        """
        Create an instance from CLI arguments.

        Args:
            args: The parser or namespace to take arguments from. Only known arguments will be
             parsed and passed to the :class:`LightningDataModule`.
            **kwargs: Additional keyword arguments that may override ones in the parser or namespace.
             These must be valid DataModule arguments.

        Example::

            parser = ArgumentParser(add_help=False)
            parser = LightningDataModule.add_argparse_args(parser)
            module = LightningDataModule.from_argparse_args(args)

        """
        if isinstance(args, ArgumentParser):
            args = cls.parse_argparser(args)
        params = vars(args)

        # we only want to pass in valid DataModule args, the rest may be user specific
        valid_kwargs = inspect.signature(cls.__init__).parameters
        datamodule_kwargs = dict((name, params[name]) for name in valid_kwargs if name in params)
        datamodule_kwargs.update(**kwargs)

        return cls(**datamodule_kwargs)

    @classmethod
    def get_init_arguments_and_types(cls) -> List[Tuple[str, Tuple, Any]]:
        r"""Scans the DataModule signature and returns argument names, types and default values.

        Returns:
            List with tuples of 3 values:
            (argument name, set with argument types, argument default value).
        """
        datamodule_default_params = inspect.signature(cls.__init__).parameters
        name_type_default = []
        for arg in datamodule_default_params:
            arg_type = datamodule_default_params[arg].annotation
            arg_default = datamodule_default_params[arg].default
            try:
                arg_types = tuple(arg_type.__args__)
            except AttributeError:
                arg_types = (arg_type, )

            name_type_default.append((arg, arg_types, arg_default))

        return name_type_default

    @classmethod
    def from_datasets(
        cls,
        train_dataset: Optional[Union[Dataset, Sequence[Dataset], Mapping[str, Dataset]]] = None,
        val_dataset: Optional[Union[Dataset, Sequence[Dataset]]] = None,
        test_dataset: Optional[Union[Dataset, Sequence[Dataset]]] = None,
        batch_size: int = 1,
        num_workers: int = 0,
    ):
        r"""
        Create an instance from torch.utils.data.Dataset.

        Args:
            train_dataset: (optional) Dataset to be used for train_dataloader()
            val_dataset: (optional) Dataset or list of Dataset to be used for val_dataloader()
            test_dataset: (optional) Dataset or list of Dataset to be used for test_dataloader()
            batch_size: Batch size to use for each dataloader. Default is 1.
            num_workers: Number of subprocesses to use for data loading. 0 means that the
                data will be loaded in the main process. Number of CPUs available.

        """

        def dataloader(ds, shuffle=False):
            return DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True,
            )

        def train_dataloader():
            if isinstance(train_dataset, Mapping):
                return {key: dataloader(ds, shuffle=True) for key, ds in train_dataset.items()}
            if isinstance(train_dataset, Sequence):
                return [dataloader(ds, shuffle=True) for ds in train_dataset]
            return dataloader(train_dataset, shuffle=True)

        def val_dataloader():
            if isinstance(val_dataset, Sequence):
                return [dataloader(ds) for ds in val_dataset]
            return dataloader(val_dataset)

        def test_dataloader():
            if isinstance(test_dataset, Sequence):
                return [dataloader(ds) for ds in test_dataset]
            return dataloader(test_dataset)

        datamodule = cls()
        if train_dataset is not None:
            datamodule.train_dataloader = train_dataloader
        if val_dataset is not None:
            datamodule.val_dataloader = val_dataloader
        if test_dataset is not None:
            datamodule.test_dataloader = test_dataloader
        return datamodule
