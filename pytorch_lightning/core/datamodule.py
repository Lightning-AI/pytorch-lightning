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
from argparse import ArgumentParser, Namespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from torch.utils.data import DataLoader, Dataset, IterableDataset

from pytorch_lightning.core.hooks import CheckpointHooks, DataHooks
from pytorch_lightning.core.mixins import HyperparametersMixin
from pytorch_lightning.utilities import rank_zero_deprecation
from pytorch_lightning.utilities.argparse import add_argparse_args, from_argparse_args, get_init_arguments_and_types


class LightningDataModule(CheckpointHooks, DataHooks, HyperparametersMixin):
    """A DataModule standardizes the training, val, test splits, data preparation and transforms. The main
    advantage is consistent data splits, data preparation and transforms across models.

    Example::

        class MyDataModule(LightningDataModule):
            def __init__(self):
                super().__init__()
            def prepare_data(self):
                # download, split, etc...
                # only called on 1 GPU/TPU in distributed
            def setup(self, stage):
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
    """

    name: str = ...

    def __init__(self, train_transforms=None, val_transforms=None, test_transforms=None, dims=None):
        super().__init__()
        if train_transforms is not None:
            rank_zero_deprecation(
                "DataModule property `train_transforms` was deprecated in v1.5 and will be removed in v1.7."
            )
        if val_transforms is not None:
            rank_zero_deprecation(
                "DataModule property `val_transforms` was deprecated in v1.5 and will be removed in v1.7."
            )
        if test_transforms is not None:
            rank_zero_deprecation(
                "DataModule property `test_transforms` was deprecated in v1.5 and will be removed in v1.7."
            )
        if dims is not None:
            rank_zero_deprecation("DataModule property `dims` was deprecated in v1.5 and will be removed in v1.7.")
        self._train_transforms = train_transforms
        self._val_transforms = val_transforms
        self._test_transforms = test_transforms
        self._dims = dims if dims is not None else ()

        # Pointer to the trainer object
        self.trainer = None

    @property
    def train_transforms(self):
        """Optional transforms (or collection of transforms) you can apply to train dataset.

        .. deprecated:: v1.5     Will be removed in v1.7.0.
        """

        rank_zero_deprecation(
            "DataModule property `train_transforms` was deprecated in v1.5 and will be removed in v1.7."
        )
        return self._train_transforms

    @train_transforms.setter
    def train_transforms(self, t):
        rank_zero_deprecation(
            "DataModule property `train_transforms` was deprecated in v1.5 and will be removed in v1.7."
        )
        self._train_transforms = t

    @property
    def val_transforms(self):
        """Optional transforms (or collection of transforms) you can apply to validation dataset.

        .. deprecated:: v1.5     Will be removed in v1.7.0.
        """

        rank_zero_deprecation(
            "DataModule property `val_transforms` was deprecated in v1.5 and will be removed in v1.7."
        )
        return self._val_transforms

    @val_transforms.setter
    def val_transforms(self, t):
        rank_zero_deprecation(
            "DataModule property `val_transforms` was deprecated in v1.5 and will be removed in v1.7."
        )
        self._val_transforms = t

    @property
    def test_transforms(self):
        """Optional transforms (or collection of transforms) you can apply to test dataset.

        .. deprecated:: v1.5     Will be removed in v1.7.0.
        """

        rank_zero_deprecation(
            "DataModule property `test_transforms` was deprecated in v1.5 and will be removed in v1.7."
        )
        return self._test_transforms

    @test_transforms.setter
    def test_transforms(self, t):
        rank_zero_deprecation(
            "DataModule property `test_transforms` was deprecated in v1.5 and will be removed in v1.7."
        )
        self._test_transforms = t

    @property
    def dims(self):
        """A tuple describing the shape of your data. Extra functionality exposed in ``size``.

        .. deprecated:: v1.5     Will be removed in v1.7.0.
        """
        rank_zero_deprecation("DataModule property `dims` was deprecated in v1.5 and will be removed in v1.7.")
        return self._dims

    @dims.setter
    def dims(self, d):
        rank_zero_deprecation("DataModule property `dims` was deprecated in v1.5 and will be removed in v1.7.")
        self._dims = d

    def size(self, dim=None) -> Union[Tuple, List[Tuple]]:
        """Return the dimension of each input either as a tuple or list of tuples. You can index this just as you
        would with a torch tensor.

        .. deprecated:: v1.5     Will be removed in v1.7.0.
        """
        rank_zero_deprecation("DataModule property `size` was deprecated in v1.5 and will be removed in v1.7.")

        if dim is not None:
            return self.dims[dim]

        return self.dims

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        """Extends existing argparse by default `LightningDataModule` attributes."""
        return add_argparse_args(cls, parent_parser, **kwargs)

    @classmethod
    def from_argparse_args(cls, args: Union[Namespace, ArgumentParser], **kwargs):
        """Create an instance from CLI arguments.

        Args:
            args: The parser or namespace to take arguments from. Only known arguments will be
                parsed and passed to the :class:`~pytorch_lightning.core.datamodule.LightningDataModule`.
            **kwargs: Additional keyword arguments that may override ones in the parser or namespace.
                These must be valid DataModule arguments.

        Example::

            parser = ArgumentParser(add_help=False)
            parser = LightningDataModule.add_argparse_args(parser)
            module = LightningDataModule.from_argparse_args(args)
        """
        return from_argparse_args(cls, args, **kwargs)

    @classmethod
    def get_init_arguments_and_types(cls) -> List[Tuple[str, Tuple, Any]]:
        r"""Scans the DataModule signature and returns argument names, types and default values.

        Returns:
            List with tuples of 3 values:
            (argument name, set with argument types, argument default value).
        """
        return get_init_arguments_and_types(cls)

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

        def dataloader(ds: Dataset, shuffle: bool = False) -> DataLoader:
            shuffle &= not isinstance(ds, IterableDataset)
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

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

    def state_dict(self) -> Dict[str, Any]:
        """Called when saving a checkpoint, implement to generate and save datamodule state.

        Returns:
            A dictionary containing datamodule state.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint, implement to reload datamodule state given datamodule state_dict.

        Args:
            state_dict: the datamodule state returned by ``state_dict``.
        """
        pass
