from abc import ABC, abstractmethod

from tests.base.dataloaders import CustomInfDataloader
from tests.base.dataloaders import CustomNotImplementedErrorDataloader
from tests.base import TrialMNIST
from torch.utils.data import DataLoader


class TrainDataloaderVariations(ABC):

    @abstractmethod
    def dataloader(self, train: bool):
        """placeholder"""

    def train_dataloader(self):
        return self.dataloader(train=True)

    def train_dataloader__infinite(self):
        return CustomInfDataloader(self.dataloader(train=True))

    def train_dataloader__long(self):
        dataset = DataLoader(TrialMNIST(download=True, num_samples=15000,
                                        digits=(0, 1, 2, 5, 8)), batch_size=32)
        return dataset

    def train_dataloader__not_implemented_error(self):
        return CustomNotImplementedErrorDataloader(self.dataloader(train=True))

    def train_dataloader__zero_length(self):
        dataloader = self.dataloader(train=True)
        dataloader.dataset.data = dataloader.dataset.data[:0]
        dataloader.dataset.targets = dataloader.dataset.targets[:0]
        return dataloader
