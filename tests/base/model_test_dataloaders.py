from abc import ABC, abstractmethod

from tests.base.dataloaders import CustomInfDataloader


class TestDataloaderVariations(ABC):

    @abstractmethod
    def dataloader(self, train: bool):
        """placeholder"""

    def test_dataloader(self):
        return self.dataloader(train=False)

    def test_dataloader__infinite(self):
        return CustomInfDataloader(self.dataloader(train=False))

    def test_dataloader__empty(self):
        return None

    def test_dataloader__multiple(self):
        return [self.dataloader(train=False), self.dataloader(train=False)]
