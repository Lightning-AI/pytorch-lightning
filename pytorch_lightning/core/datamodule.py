import inspect
from abc import abstractmethod
from argparse import ArgumentParser, Namespace
from typing import Any, List, Tuple, Union

from torch.utils.data import DataLoader

from pytorch_lightning.utilities import parsing, rank_zero_only, rank_zero_warn


class _DataModuleWrapper(type):
    def __call__(cls, *args, **kwargs):
        """A wrapper for LightningDataModule that:

            1. Runs user defined subclass's __init__
            2. Assures prepare_data() runs on rank 0
        """

        # Get instance of LightningDataModule by mocking its __init__ via __call__
        obj = type.__call__(cls, *args, **kwargs)

        # Wrap instance's prepare_data function with rank_zero_only and reassign to instance
        obj.prepare_data = rank_zero_only(obj.prepare_data)

        return obj


class LightningDataModule(object, metaclass=_DataModuleWrapper):  # pragma: no cover
    """
    A DataModule standardizes the training, val, test splits, data preparation and transforms.
    The main advantage is consistent data splits and transforms across models.

    Example::

        class MyDataModule(LightningDataModule):
            def __init__(self):
                super().__init__()
            def prepare_data(self):
                # download, split, etc...
                # only called on rank 0
            def setup(self):
                # make assignments here
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

    A DataModule implements 5 key methods
    1. **prepare_data** (things to do on 1 GPU not on every GPU in distributed mode)
    2. **setup**  (things to do on every GPU in distributed mode)
    2. **train_dataloader** the training dataloader.
    3. **val_dataloader** the val dataloader.
    4. **test_dataloader** the test dataloader.
    This allows you to share a full dataset without explaining what the splits, transforms or download
    process is.
    """

    name: str = ...

    def __init__(
        self, train_transforms=None, val_transforms=None, test_transforms=None,
    ):
        super().__init__()
        self._train_transforms = train_transforms
        self._val_transforms = val_transforms
        self._test_transforms = test_transforms
        self.dims = ()

    @property
    def train_transforms(self):
        """
        Optional transforms you can apply to train dataset
        """
        return self._train_transforms

    @train_transforms.setter
    def train_transforms(self, t):
        self._train_transforms = t

    @property
    def val_transforms(self):
        """
        Optional transforms you can apply to validation dataset
        """
        return self._val_transforms

    @val_transforms.setter
    def val_transforms(self, t):
        self._val_transforms = t

    @property
    def test_transforms(self):
        """
        Optional transforms you can apply to test dataset
        """
        return self._test_transforms

    @test_transforms.setter
    def test_transforms(self, t):
        self._test_transforms = t

    def size(self, dim=None) -> Union[Tuple, int]:
        """
        Return the dimension of each input either as a tuple or list of tuples.
        """

        if dim is not None:
            return self.dims[dim]

        return self.dims

    @abstractmethod
    def prepare_data(self, *args, **kwargs):
        """
        Use this to download and prepare data.
        In distributed (GPU, TPU), this will only be called once.

        .. warning:: Do not assign anything to the datamodule in this step since this will only be called on 1 GPU.

        Pseudocode::

            dm.prepare_data()
            dm.setup()

        Example::

            def prepare_data(self):
                download_imagenet()
                clean_imagenet()
                cache_imagenet()
        """

    @abstractmethod
    def setup(self, *args, **kwargs):
        """
        Use this to load your data from file, split it, etc. You are safe to make state assignments here.
        This hook is called on every process when using DDP.

        Example::

            def setup(self):
                data = load_data(...)
                self.train_ds, self.val_ds, self.test_ds = split_data(data)
        """

    @abstractmethod
    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        """
        Implement a PyTorch DataLoader for training.
        Return:
            Single PyTorch :class:`~torch.utils.data.DataLoader`.
        Note:
            Lightning adds the correct sampler for distributed and arbitrary hardware.
            There is no need to set it yourself.

        Example::

            def train_dataloader(self):
                dataset = MNIST(root=PATH, train=True, transform=transforms.ToTensor(), download=False)
                loader = torch.utils.data.DataLoader(dataset=dataset)
                return loader

        """
        rank_zero_warn('`train_dataloader` must be implemented to be used with the Lightning Trainer')

    @abstractmethod
    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        r"""
        Implement a PyTorch DataLoader for training.
        Return:
            Single PyTorch :class:`~torch.utils.data.DataLoader`.
        Note:
            Lightning adds the correct sampler for distributed and arbitrary hardware.
            There is no need to set it yourself.
        Note:
            You can also return a list of DataLoaders

        Example::

            def val_dataloader(self):
                dataset = MNIST(root=PATH, train=False, transform=transforms.ToTensor(), download=False)
                loader = torch.utils.data.DataLoader(dataset=dataset, shuffle=False)
                return loader
        """

    @abstractmethod
    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        r"""
        Implement a PyTorch DataLoader for training.
        Return:
            Single PyTorch :class:`~torch.utils.data.DataLoader`.
        Note:
            Lightning adds the correct sampler for distributed and arbitrary hardware.
            There is no need to set it yourself.
        Note:
            You can also return a list of DataLoaders

        Example::

            def test_dataloader(self):
                dataset = MNIST(root=PATH, train=False, transform=transforms.ToTensor(), download=False)
                loader = torch.utils.data.DataLoader(dataset=dataset, shuffle=False)
                return loader
        """

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser) -> ArgumentParser:
        r"""Extends existing argparse by default `LightningDataModule` attributes.
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False,)
        added_args = [x.dest for x in parser._actions]

        blacklist = ['kwargs']
        depr_arg_names = blacklist + added_args
        depr_arg_names = set(depr_arg_names)

        allowed_types = (str, float, int, bool)

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
                arg_kwargs.update(nargs="?")
                # if the only arg type is bool
                if len(arg_types) == 1:
                    # redefine the type for ArgParser needed
                    def use_type(x):
                        return bool(parsing.str_to_bool(x))

                else:
                    # filter out the bool as we need to use more general
                    use_type = [at for at in arg_types if at is not bool][0]
            else:
                use_type = arg_types[0]

            if arg_default == inspect._empty:
                arg_default = None

            parser.add_argument(
                f'--{arg}',
                dest=arg,
                default=arg_default,
                type=use_type,
                help=f'autogenerated by plb.{cls.__name__}',
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
                arg_types = (arg_type,)

            name_type_default.append((arg, arg_types, arg_default))

        return name_type_default
