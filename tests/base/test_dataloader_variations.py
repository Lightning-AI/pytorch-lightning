from abc import ABC, abstractmethod


class TestDataloaderVariationsMixin(ABC):

    @abstractmethod
    def dataloader(self, train):
        """placeholder"""
        pass

    def test_dataloader(self):
        return self.dataloader(
            train=False,
        )
