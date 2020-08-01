from torch.utils.data import random_split, DataLoader

from pytorch_lightning.core.datamodule import LightningDataModule
from tests.base.datasets import TrialMNIST


class TrialMNISTDataModule(LightningDataModule):
    def __init__(self, data_dir: str = './'):
        super().__init__()
        self.data_dir = data_dir

    def prepare_data(self):
        TrialMNIST(self.data_dir, train=True, download=True)
        TrialMNIST(self.data_dir, train=False, download=True)

    def setup(self):
        mnist_full = TrialMNIST(root=self.data_dir, train=True, num_samples=64, download=True)
        self.mnist_train, self.mnist_val = random_split(mnist_full, [128, 64])
        self.dims = tuple(self.mnist_train[0][0].shape)
        self.mnist_test = TrialMNIST(root=self.data_dir, train=False, num_samples=32, download=True)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32)
