import os
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

    resources = [
        "https://pl-public-data.s3.amazonaws.com/MNIST/processed/training.pt",
        "https://pl-public-data.s3.amazonaws.com/MNIST/processed/test.pt",
    ]

    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, root, train=True, normalize=(0.5, 1.0), download=False):
        super(MNIST, self).__init__()
        self.root = root
        self.train = train  # training set or test set
        self.normalize = normalize

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index):
        img, target = self.data[index].unsqueeze(0), int(self.targets[index])

        if self.normalize is not None:
            img = normalize_tensor(img, mean=self.normalize[0], std=self.normalize[1])

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    def _check_exists(self):
        train_file = os.path.join(self.processed_folder, self.training_file)
        test_file = os.path.join(self.processed_folder, self.test_file)
        return os.path.exists(train_file) and os.path.exists(test_file)

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        os.makedirs(self.processed_folder, exist_ok=True)

        for url in self.resources:
            # TODO: use logging
            print('Downloading ' + url)
            fpath = os.path.join(self.processed_folder, os.path.basename(url))
            urllib.request.urlretrieve(url, fpath)


def normalize_tensor(tensor, mean=0.0, std=1.0):
    tensor = tensor.clone()
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    tensor.sub_(mean).div_(std)
    return tensor
