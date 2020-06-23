import logging
import os
import pickle
import tarfile
import urllib.request
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Sequence
from urllib.error import HTTPError

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from tests import PACKAGE_ROOT

#: local path to test datasets
PATH_DATASETS = os.path.join(PACKAGE_ROOT, 'Datasets')


class LightDataset(ABC, Dataset):

    data: torch.Tensor
    targets: torch.Tensor
    normalize: tuple
    root_path: str
    cache_folder_name: str
    DATASET_NAME = 'light'

    def __len__(self) -> int:
        return len(self.data)

    @abstractmethod
    def normalize_tensor(self, img: Tensor, mean, std) -> Tensor:
        """Normalise image."""

    @property
    def cached_folder_path(self) -> str:
        return os.path.join(self.root_path, self.DATASET_NAME, self.cache_folder_name)

    @staticmethod
    def _prepare_subset(
            full_data: torch.Tensor,
            full_targets: torch.Tensor,
            num_samples: int,
            labels: Sequence
    ) -> Tuple[Tensor, Tensor]:
        """Prepare a subset of a common dataset."""
        classes = {d: 0 for d in labels}
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

    def _download_from_url(self, base_url: str, data_folder: str, file_name: str):
        url = os.path.join(base_url, file_name)
        logging.info(f'Downloading {url}')
        fpath = os.path.join(data_folder, file_name)
        try:
            urllib.request.urlretrieve(url, fpath)
        except HTTPError:
            raise RuntimeError(f'Failed download from {url}')


class MNIST(LightDataset):
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
        >>> data, label = dataset[0]
        >>> data.shape
        torch.Size([1, 28, 28])
        >>> label
        5
    """

    BASE_URL = "https://pl-public-data.s3.amazonaws.com/MNIST/processed"
    TRAIN_FILE_NAME = 'training.pt'
    TEST_FILE_NAME = 'test.pt'
    cache_folder_name = 'complete'
    DATASET_NAME = 'MNIST'

    def __init__(
            self,
            root: str = PATH_DATASETS,
            train: bool = True,
            normalize: tuple = (0.5, 1.0),
            download: bool = True
    ):
        super().__init__()
        self.root_path = root
        self.train = train  # training set or test set
        self.normalize = normalize

        os.makedirs(self.cached_folder_path, exist_ok=True)
        self.prepare_data(download)

        if not self._check_exists(self.cached_folder_path):
            raise RuntimeError('Dataset not found.')

        data_file = self.TRAIN_FILE_NAME if self.train else self.TEST_FILE_NAME
        self.data, self.targets = torch.load(os.path.join(self.cached_folder_path, data_file))

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        img = self.data[idx].float().unsqueeze(0)
        target = int(self.targets[idx])

        if self.normalize is not None:
            img = self.normalize_tensor(img, mean=self.normalize[0], std=self.normalize[1])

        return img, target

    def normalize_tensor(self, tensor: Tensor, mean: float = 0.0, std: float = 1.0) -> Tensor:
        # tensor = tensor.clone()
        mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
        std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
        tensor.sub_(mean).div_(std)
        return tensor

    @classmethod
    def _check_exists(cls, data_folder: str) -> bool:
        return all(os.path.isfile(os.path.join(data_folder, fname))
                   for fname in (cls.TRAIN_FILE_NAME, cls.TEST_FILE_NAME))

    def prepare_data(self, download: bool):
        if download:
            self._download(self.cached_folder_path)

    def _download(self, data_folder: str) -> None:
        """Download the MNIST data if it doesn't exist in cached_folder_path already."""

        if self._check_exists(data_folder):
            return

        for fname in (self.TRAIN_FILE_NAME, self.TEST_FILE_NAME):
            self._download_from_url(self.BASE_URL, data_folder, fname)


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
        >>> data, label = dataset[0]
        >>> data.shape
        torch.Size([1, 28, 28])
    """

    def __init__(
            self,
            root: str = PATH_DATASETS,
            train: bool = True,
            normalize: tuple = (0.5, 1.0),
            download: bool = False,
            num_samples: int = 100,
            digits: Optional[Sequence] = (0, 1, 2)
    ):

        # number of examples per class
        self.num_samples = num_samples
        # take just a subset of MNIST dataset
        self.digits = digits if digits else list(range(10))

        self.cache_folder_name = f'digits-{"-".join(str(d) for d in sorted(self.digits))}_nb-{self.num_samples}'

        super().__init__(
            root,
            train=train,
            normalize=normalize,
            download=download
        )

    def prepare_data(self, download: bool) -> None:
        super().prepare_data(download)

        for fname in (self.TRAIN_FILE_NAME, self.TEST_FILE_NAME):
            path_fname = os.path.join(super().cached_folder_path, fname)
            assert os.path.isfile(path_fname), 'Missing cached file: %s' % path_fname
            data, targets = torch.load(path_fname)
            data, targets = self._prepare_subset(data, targets, self.num_samples, self.digits)
            torch.save((data, targets), os.path.join(self.cached_folder_path, fname))


class CIFAR10(LightDataset):
    """
    Customized `CIFAR10 <http://www.cs.toronto.edu/~kriz/cifar.html>`_ dataset for testing Pytorch Lightning
    without the torchvision dependency.

    Part of the code was copied from
    https://github.com/pytorch/vision/blob/build/v0.5.0/torchvision/datasets/

    Args:
        root: Root directory of dataset where ``CIFAR10/processed/training.pt``
            and  ``CIFAR10/processed/test.pt`` exist.
        train: If ``True``, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        normalize: mean and std deviation of the MNIST dataset.
        download: If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    .. todo:: normalize per color channel

    Examples:

        >>> dataset = CIFAR10(download=True, normalize=(0., 1.))
        >>> len(dataset)
        50000
        >>> torch.bincount(dataset.targets)
        tensor([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000])
        >>> data, label = dataset[0]
        >>> data.shape
        torch.Size([3, 32, 32])
        >>> label
        6
    """

    BASE_URL = "https://www.cs.toronto.edu/~kriz/"
    FILE_NAME = 'cifar-10-python.tar.gz'
    cache_folder_name = 'complete'
    TRAIN_FILE_NAME = 'training.pt'
    TEST_FILE_NAME = 'test.pt'
    DATASET_NAME = 'CIFAR10'

    def __init__(
            self,
            root: str = PATH_DATASETS,
            train: bool = True,
            normalize: tuple = (0.5, 1.0),
            download: bool = True
    ):
        super().__init__()
        self.root_path = root
        self.train = train  # training set or test set
        self.normalize = normalize

        os.makedirs(self.cached_folder_path, exist_ok=True)
        self.prepare_data(download)

        if not self._check_exists(self.cached_folder_path, (self.TRAIN_FILE_NAME, self.TEST_FILE_NAME)):
            raise RuntimeError('Dataset not found.')

        data_file = self.TRAIN_FILE_NAME if self.train else self.TEST_FILE_NAME
        self.data, self.targets = torch.load(os.path.join(self.cached_folder_path, data_file))

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        img = self.data[idx].float().reshape(3, 32, 32)
        target = int(self.targets[idx])

        if self.normalize is not None:
            img = self.normalize_tensor(img, mean=self.normalize[0], std=self.normalize[1])

        return img, target

    @classmethod
    def _check_exists(cls, data_folder: str, file_names: Sequence[str]) -> bool:
        if isinstance(file_names, str):
            file_names = [file_names]
        return all(os.path.isfile(os.path.join(data_folder, fname))
                   for fname in file_names)

    def normalize_tensor(self, tensor: Tensor, mean: float = 0.0, std: float = 1.0) -> Tensor:
        # tensor = tensor.clone()
        # todo: normalize per color channel
        mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
        std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
        tensor.sub_(mean).div_(std)
        return tensor

    def _unpickle(self, path_folder: str, file_name: str) -> Tuple[Tensor, Tensor]:
        with open(os.path.join(path_folder, file_name), 'rb') as fo:
            pkl = pickle.load(fo, encoding='bytes')
        return torch.tensor(pkl[b'data']), torch.tensor(pkl[b'labels'])

    def _extract_archive_save_torch(self, download_path):
        # extract achieve
        with tarfile.open(os.path.join(download_path, self.FILE_NAME), 'r:gz') as tar:
            tar.extractall(path=download_path)
        # this is internal path in the archive
        path_content = os.path.join(download_path, 'cifar-10-batches-py')

        # load Test and save as PT
        torch.save(self._unpickle(path_content, 'test_batch'),
                   os.path.join(self.cached_folder_path, self.TEST_FILE_NAME))
        # load Train and save as PT
        data, labels = [], []
        for i in range(5):
            fname = f'data_batch_{i + 1}'
            _data, _labels = self._unpickle(path_content, fname)
            data.append(_data)
            labels.append(_labels)
        # stach all to one
        data = torch.cat(data, dim=0)
        labels = torch.cat(labels, dim=0)
        # and save as PT
        torch.save((data, labels), os.path.join(self.cached_folder_path, self.TRAIN_FILE_NAME))

    def prepare_data(self, download: bool):
        if self._check_exists(self.cached_folder_path, (self.TRAIN_FILE_NAME, self.TEST_FILE_NAME)):
            return

        base_path = os.path.join(self.root_path, self.DATASET_NAME)
        if download:
            self.download(base_path)
        self._extract_archive_save_torch(base_path)

    def download(self, data_folder: str) -> None:
        """Download the data if it doesn't exist in cached_folder_path already."""
        if self._check_exists(data_folder, self.FILE_NAME):
            return
        self._download_from_url(self.BASE_URL, data_folder, self.FILE_NAME)


class TrialCIFAR10(CIFAR10):
    """
    Customized `CIFAR10 <http://www.cs.toronto.edu/~kriz/cifar.html>`_ dataset for testing Pytorch Lightning
    without the torchvision dependency.

    Args:
        root: Root directory of dataset where ``CIFAR10/processed/training.pt``
            and  ``CIFAR10/processed/test.pt`` exist.
        train: If ``True``, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        normalize: mean and std deviation of the MNIST dataset.
        download: If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        num_samples: number of examples per selected class/digit
        labels: list selected MNIST digits/classes

    Examples:

        >>> dataset = TrialCIFAR10(download=True, normalize=(0., 1.), num_samples=150)
        >>> len(dataset)
        450
        >>> sorted(set([d.item() for d in dataset.targets]))
        [0, 1, 2]
        >>> torch.bincount(dataset.targets)
        tensor([150, 150, 150])
        >>> data, label = dataset[0]
        >>> data.shape
        torch.Size([3, 32, 32])
    """
    def __init__(
            self,
            root: str = PATH_DATASETS,
            train: bool = True,
            normalize: tuple = (0.5, 1.0),
            download: bool = False,
            num_samples: int = 100,
            labels: Optional[Sequence] = (0, 1, 2)
    ):
        # number of examples per class
        self.num_samples = num_samples
        # take just a subset of CIFAR dataset
        self.labels = labels if labels else list(range(10))

        self.cache_folder_name = f'labels-{"-".join(str(d) for d in sorted(self.labels))}_nb-{self.num_samples}'

        super().__init__(
            root,
            train=train,
            normalize=normalize,
            download=download
        )

    def prepare_data(self, download: bool) -> None:
        super().prepare_data(download)

        for fname in (self.TRAIN_FILE_NAME, self.TEST_FILE_NAME):
            path_fname = os.path.join(super().cached_folder_path, fname)
            assert os.path.isfile(path_fname), 'Missing cached file: %s' % path_fname
            data, targets = torch.load(path_fname)
            data, targets = self._prepare_subset(data, targets, self.num_samples, self.labels)
            torch.save((data, targets), os.path.join(self.cached_folder_path, fname))


class AverageDataset(Dataset):

    def __init__(self, dataset_len=300, sequence_len=100):
        self.dataset_len = dataset_len
        self.sequence_len = sequence_len
        self.input_seq = torch.randn(dataset_len, sequence_len, 10)
        top, bottom = self.input_seq.chunk(2, -1)
        self.output_seq = top + bottom.roll(shifts=1, dims=-1)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, item):
        return self.input_seq[item], self.output_seq[item]
