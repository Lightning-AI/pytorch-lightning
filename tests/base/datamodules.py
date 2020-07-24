from torch.utils.data import random_split, DataLoader

from pytorch_lightning import LightningDataModule
from tests.base.datasets import MNIST


class MNISTDataModule(LightningDataModule):

    def __init__(self, data_dir: str = './'):
        super(MNISTDataModule, self).__init__()
        self.data_dir = data_dir

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)
    
    def setup(self):
        mnist_full = MNIST(self.data_dir, train=True, download=False)
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
        self.dims = tuple(self.mnist_train[0][0].shape)
        self.mnist_test = MNIST(self.data_dir, train=False, download=False)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32)
