import logging
import os
import urllib.request
from typing import Tuple, Optional, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from tests import PACKAGE_ROOT

#: local path to test datasets
PATH_DATASETS = os.path.join(PACKAGE_ROOT, 'Datasets')


class MNIST(Dataset):
    """
    Customized `MNIST <http://yann.lecun.com/exdb/mnist/>`_ dataset for testing Pytorch Lightning
    without the torchvision dependency.

    Part of the code was copied from
    https://github.com/pytorch/vision/blob/build/v0.5.0/torchvision/datasets/mnist.py

    Args:
        root: Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train: If ``True``, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        normalize: mean and std deviation of the MNIST dataset.
        download: If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    Examples:
        >>> dataset = MNIST(download=True)
        >>> len(dataset)
        60000
        >>> torch.bincount(dataset.targets)
        tensor([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949])
    """

    RESOURCES = (
        "https://pl-public-data.s3.amazonaws.com/MNIST/processed/training.pt",
        "https://pl-public-data.s3.amazonaws.com/MNIST/processed/test.pt",
    )

    TRAIN_FILE_NAME = 'training.pt'
    TEST_FILE_NAME = 'test.pt'
    cache_folder_name = 'complete'

    def __init__(self, root: str = PATH_DATASETS, train: bool = True,
                 normalize: tuple = (0.5, 1.0), download: bool = True):
        super().__init__()
        self.root = root
        self.train = train  # training set or test set
        self.normalize = normalize

        self.prepare_data(download)

        if not self._check_exists(self.cached_folder_path):
            raise RuntimeError('Dataset not found.')

        data_file = self.TRAIN_FILE_NAME if self.train else self.TEST_FILE_NAME
        self.data, self.targets = torch.load(os.path.join(self.cached_folder_path, data_file))

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        img = self.data[idx].float().unsqueeze(0)
        target = int(self.targets[idx])

        if self.normalize is not None:
            img = normalize_tensor(img, mean=self.normalize[0], std=self.normalize[1])

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def cached_folder_path(self) -> str:
        return os.path.join(self.root, 'MNIST', self.cache_folder_name)

    def _check_exists(self, data_folder: str) -> bool:
        existing = True
        for fname in (self.TRAIN_FILE_NAME, self.TEST_FILE_NAME):
            existing = existing and os.path.isfile(os.path.join(data_folder, fname))
        return existing

    def prepare_data(self, download: bool):
        if download:
            self._download(self.cached_folder_path)

    def _download(self, data_folder: str) -> None:
        """Download the MNIST data if it doesn't exist in cached_folder_path already."""

        if self._check_exists(data_folder):
            return

        os.makedirs(data_folder, exist_ok=True)

        for url in self.RESOURCES:
            logging.info(f'Downloading {url}')
            fpath = os.path.join(data_folder, os.path.basename(url))
            urllib.request.urlretrieve(url, fpath)


def normalize_tensor(tensor: Tensor, mean: float = 0.0, std: float = 1.0) -> Tensor:
    tensor = tensor.clone()
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    tensor.sub_(mean).div_(std)
    return tensor


class TrialMNIST(MNIST):
    """Constrain image dataset

    Args:
        root: Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train: If ``True``, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        normalize: mean and std deviation of the MNIST dataset.
        download: If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        num_samples: number of examples per selected class/digit
        digits: list selected MNIST digits/classes

    Examples:
        >>> dataset = TrialMNIST(download=True)
        >>> len(dataset)
        300
        >>> sorted(set([d.item() for d in dataset.targets]))
        [0, 1, 2]
        >>> torch.bincount(dataset.targets)
        tensor([100, 100, 100])
    """

    def __init__(self, root: str = PATH_DATASETS, train: bool = True,
                 normalize: tuple = (0.5, 1.0), download: bool = False,
                 num_samples: int = 100, digits: Optional[Sequence] = (0, 1, 2)):

        # number of examples per class
        self.num_samples = num_samples
        # take just a subset of MNIST dataset
        self.digits = digits if digits else list(range(10))

        self.cache_folder_name = 'digits-' + '-'.join(str(d) for d in sorted(self.digits)) \
                                 + f'_nb-{self.num_samples}'

        super().__init__(
            root,
            train=train,
            normalize=normalize,
            download=download
        )

    @staticmethod
    def _prepare_subset(full_data: torch.Tensor, full_targets: torch.Tensor,
                        num_samples: int, digits: Sequence):
        classes = {d: 0 for d in digits}
        indexes = []
        for idx, target in enumerate(full_targets):
            label = target.item()
            if classes.get(label, float('inf')) >= num_samples:
                continue
            indexes.append(idx)
            classes[label] += 1
            if all(classes[k] >= num_samples for k in classes):
                break
        data = full_data[indexes]
        targets = full_targets[indexes]
        return data, targets

    def prepare_data(self, download: bool) -> None:
        if self._check_exists(self.cached_folder_path):
            return
        if download:
            self._download(super().cached_folder_path)

        for fname in (self.TRAIN_FILE_NAME, self.TEST_FILE_NAME):
            path_fname = os.path.join(super().cached_folder_path, fname)
            assert os.path.isfile(path_fname), 'Missing cached file: %s' % path_fname
            data, targets = torch.load(path_fname)
            data, targets = self._prepare_subset(data, targets, self.num_samples, self.digits)
            torch.save((data, targets), os.path.join(self.cached_folder_path, fname))
