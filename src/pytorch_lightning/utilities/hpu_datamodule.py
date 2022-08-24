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

from typing import Any, Optional

import pytorch_lightning as pl
from pytorch_lightning.utilities import _HPU_AVAILABLE
from pytorch_lightning.utilities.imports import _TORCHVISION_AVAILABLE

if _TORCHVISION_AVAILABLE:
    import torchvision.datasets
    from torchvision import transforms as transform_lib

if _HPU_AVAILABLE:
    try:
        import habana_dataloader
    except ImportError:
        raise ModuleNotFoundError("habana_dataloader package is not installed.")

_DATASETS_PATH = "./data"


def load_data(traindir, valdir, train_transforms, val_transforms):  # type: ignore[no-untyped-def]

    # check supported transforms
    if train_transforms is not None:
        for t in train_transforms:
            if (
                isinstance(t, transform_lib.RandomResizedCrop)
                or isinstance(t, transform_lib.RandomHorizontalFlip)
                or isinstance(t, transform_lib.ToTensor)
                or isinstance(t, transform_lib.Normalize)
            ):

                continue
            else:
                raise ValueError("Unsupported train transform: " + str(type(t)))

        train_transforms = transform_lib.Compose(train_transforms)

    if val_transforms is not None:
        for t in val_transforms:
            if (
                isinstance(t, transform_lib.Resize)
                or isinstance(t, transform_lib.CenterCrop)
                or isinstance(t, transform_lib.ToTensor)
                or isinstance(t, transform_lib.Normalize)
            ):

                continue
            else:
                raise ValueError("Unsupported val transform: " + str(type(t)))

        val_transforms = transform_lib.Compose(val_transforms)

    if "imagenet" not in traindir.lower() and "ilsvrc2012" not in traindir.lower():
        raise ValueError("Habana dataloader only supports Imagenet dataset")

    # Data loading code
    dataset_train = torchvision.datasets.ImageFolder(traindir, train_transforms)

    dataset_val = torchvision.datasets.ImageFolder(valdir, val_transforms)

    return dataset_train, dataset_val


class HPUDataModule(pl.LightningDataModule):

    name = "hpu-dataset"

    def __init__(
        self,
        train_dir: str = _DATASETS_PATH,
        val_dir: str = _DATASETS_PATH,
        num_workers: int = 8,
        normalize: bool = False,
        seed: int = 42,
        batch_size: int = 32,
        train_transforms: Any = None,
        val_transforms: Any = None,
        pin_memory: bool = True,
        shuffle: bool = False,
        drop_last: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.train_dir = train_dir
        self.val_dir = val_dir
        self.num_workers = num_workers
        self.normalize = normalize
        self.seed = seed
        self.batch_size = batch_size
        self.train_transform = train_transforms
        self.val_transform = val_transforms
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.data_loader_type = habana_dataloader.HabanaDataLoader

    def setup(self, stage: Optional[str] = None):  # type: ignore[no-untyped-def]

        if not _TORCHVISION_AVAILABLE:
            raise ValueError("torchvision transforms not available")

        if self.shuffle is True:
            raise ValueError("HabanaDataLoader does not support shuffle=True")

        if self.pin_memory is False:
            raise ValueError("HabanaDataLoader only supports pin_memory=True")

        if self.num_workers != 8:
            raise ValueError("HabanaDataLoader only supports num_workers as 8")

        self.dataset_train, self.dataset_val = load_data(
            self.train_dir, self.val_dir, self.train_transform, self.val_transform
        )

        dataset = self.dataset_train if stage == "fit" else self.dataset_val
        if dataset is None:
            raise TypeError("Error creating dataset")

    def train_dataloader(self):  # type: ignore[no-untyped-def]
        """train set removes a subset to use for validation."""
        loader = self.data_loader_type(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )
        return loader

    def val_dataloader(self):  # type: ignore[no-untyped-def]
        """val set uses a subset of the training set for validation."""
        loader = self.data_loader_type(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )
        return loader

    def test_dataloader(self):  # type: ignore[no-untyped-def]
        """test set uses the test split."""
        loader = self.data_loader_type(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )
        return loader
