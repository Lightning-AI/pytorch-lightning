from abc import ABC, abstractmethod


class EvalDataloaderPool(ABC):

    @abstractmethod
    def dataloader(self, train):
        """placeholder"""

    def test_dataloader(self):
        return self.dataloader(
            train=False,
        )


class ValDataloaderPool(ABC):

    @abstractmethod
    def dataloader(self, train):
        """placeholder"""
        pass

    def val_dataloader(self):
        return self.dataloader(
            train=False,
        )


class TrainDataloaderPool(ABC):

    @abstractmethod
    def dataloader(self, train):
        """placeholder"""
        pass

    def train_dataloader(self):
        return self.dataloader(
            train=True,
        )
