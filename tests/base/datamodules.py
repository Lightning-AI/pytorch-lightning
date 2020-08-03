from torch.utils.data import random_split, DataLoader

from pytorch_lightning.core.datamodule import LightningDataModule
from tests.base.datasets import TrialMNIST


class TrialMNISTDataModule(LightningDataModule):

    def __init__(self, data_dir: str = './'):
        super().__init__()
        self.data_dir = data_dir
        self.non_picklable = None

    def prepare_data(self):
        TrialMNIST(self.data_dir, train=True, download=True)
        TrialMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str = None):

        if stage == 'fit' or stage is None:
            mnist_full = TrialMNIST(root=self.data_dir, train=True, num_samples=64, download=True)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [128, 64])
            self.dims = self.mnist_train[0][0].shape

        if stage == 'test' or stage is None:
            self.mnist_test = TrialMNIST(root=self.data_dir, train=False, num_samples=32, download=True)
            self.dims = getattr(self, 'dims', self.mnist_test[0][0].shape)

        self.non_picklable = lambda x: x**2

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32)
