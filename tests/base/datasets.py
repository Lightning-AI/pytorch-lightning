import os
import logging
import urllib.request

import torch
from torch.utils.data import Dataset


class MNIST(Dataset):
    """
    Customized `MNIST <http://yann.lecun.com/exdb/mnist/>`_ dataset for testing Pytorch Lightning
    without the torchvision dependency.

    Part of the code was copied from
    https://github.com/pytorch/vision/blob/build/v0.5.0/torchvision/datasets/mnist.py

    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    RESOURCES = (
        "https://pl-public-data.s3.amazonaws.com/MNIST/processed/training.pt",
        "https://pl-public-data.s3.amazonaws.com/MNIST/processed/test.pt",
    )

    TRAIN_FILE_NAME = 'training.pt'
    TEST_FILE_NAME = 'test.pt'

    def __init__(self, root, train=True, normalize=(0.5, 1.0), download=False):
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

    def __getitem__(self, index):
        img, target = self.data[index].float().unsqueeze(0), int(self.targets[index])

        if self.normalize is not None:
            img = normalize_tensor(img, mean=self.normalize[0], std=self.normalize[1])

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'MNIST', 'processed')

    def _check_exists(self):
        train_file = os.path.join(self.processed_folder, self.TRAIN_FILE_NAME)
        test_file = os.path.join(self.processed_folder, self.TEST_FILE_NAME)
        return os.path.exists(train_file) and os.path.exists(test_file)

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        os.makedirs(self.processed_folder, exist_ok=True)

        for url in self.RESOURCES:
            logging.info(f'Downloading {url}')
            fpath = os.path.join(self.processed_folder, os.path.basename(url))
            urllib.request.urlretrieve(url, fpath)


def normalize_tensor(tensor, mean=0.0, std=1.0):
    tensor = tensor.clone()
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    tensor.sub_(mean).div_(std)
    return tensor


class TestingMNIST(MNIST):

    def __init__(self, root, train=True, normalize=None, download=False, num_samples=8000):
        super().__init__(
            root,
            train=train,
            normalize=normalize,
            download=download
        )
        # take just a subset of MNIST dataset
        self.data = self.data[:num_samples]
        self.targets = self.targets[:num_samples]
