from abc import ABC, abstractmethod

from tests.base.mixins import CustomInfDataloader


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
