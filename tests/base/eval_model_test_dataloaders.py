from abc import ABC, abstractmethod


class TestDataloaderVariations(ABC):

    @abstractmethod
    def dataloader(self, train: bool):
        """placeholder"""

    def test_dataloader(self):
        return self.dataloader(train=False)
