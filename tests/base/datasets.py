import itertools
import logging
import os
import urllib.request
from typing import Tuple, Optional, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset


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
    """

    RESOURCES = (
        "https://pl-public-data.s3.amazonaws.com/MNIST/processed/training.pt",
        "https://pl-public-data.s3.amazonaws.com/MNIST/processed/test.pt",
    )

    TRAIN_FILE_NAME = 'training.pt'
    TEST_FILE_NAME = 'test.pt'

    def __init__(self, root: str, train: bool = True, normalize: tuple = (0.5, 1.0), download: bool = False):
        super(MNIST, self).__init__()
        self.root = root
        self.train = train  # training set or test set
        self.normalize = normalize

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        data_file = self.TRAIN_FILE_NAME if self.train else self.TEST_FILE_NAME
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        img = self.data[idx].float().unsqueeze(0)
        target = int(self.targets[idx])

        if self.normalize is not None:
            img = normalize_tensor(img, mean=self.normalize[0], std=self.normalize[1])

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, 'MNIST', 'processed')

    def _check_exists(self) -> bool:
        train_file = os.path.join(self.processed_folder, self.TRAIN_FILE_NAME)
        test_file = os.path.join(self.processed_folder, self.TEST_FILE_NAME)
        return os.path.isfile(train_file) and os.path.isfile(test_file)

    def download(self) -> None:
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        os.makedirs(self.processed_folder, exist_ok=True)

        for url in self.RESOURCES:
            logging.info(f'Downloading {url}')
            fpath = os.path.join(self.processed_folder, os.path.basename(url))
            urllib.request.urlretrieve(url, fpath)


def normalize_tensor(tensor: Tensor, mean: float = 0.0, std: float = 1.0) -> Tensor:
    tensor = tensor.clone()
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    tensor.sub_(mean).div_(std)
    return tensor


class TestingMNIST(MNIST):
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
        num_samples: number of examples per seelcted class/digget
        digits: list selected MNIST digits/classes

    Examples:
        >>> dataset = TestingMNIST(os.getcwd(), download=True)
        >>> len(dataset)
        1200
    """

    def __init__(self, root: str, train: bool = True, normalize: tuple = (0.5, 1.0), download: bool = False,
                 num_samples: int = 300, digits: Optional[Sequence] = (0, 1, 2, 3)):
        super().__init__(
            root,
            train=train,
            normalize=normalize,
            download=download
        )
        # take just a subset of MNIST dataset
        if not digits:
            digits = list(range(10))
        if (num_samples * len(digits)) >= 60000:
            # asking for complete dataset
            return
        classes = {d: 0 for d in digits}
        indexes = []
        for idx, target in enumerate(self.targets):
            label = target.item()
            if classes.get(label, float('inf')) >= num_samples:
                continue
            indexes.append(idx)
            classes[label] += 1
            if all(classes[k] >= num_samples for k in classes):
                break
        self.data = self.data[indexes]
        self.targets = self.targets[indexes]
