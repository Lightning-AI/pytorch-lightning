from abc import ABC, abstractmethod

from tests.base.dataloaders import CustomInfDataloader
from tests.base.dataloaders import CustomNotImplementedErrorDataloader


class TrainDataloaderVariations(ABC):

    @abstractmethod
    def dataloader(self, train: bool):
        """placeholder"""

    def train_dataloader(self):
        return self.dataloader(train=True)

    def train_dataloader__infinite(self):
        return CustomInfDataloader(self.dataloader(train=True))

    def train_dataloader__not_implemented_error(self):
        return CustomNotImplementedErrorDataloader(self.dataloader(train=True))

    def train_dataloader__zero_length(self):
        dataloader = self.dataloader(train=True)
        dataloader.dataset.data = dataloader.dataset.data[:0]
        dataloader.dataset.targets = dataloader.dataset.targets[:0]
        return dataloader
