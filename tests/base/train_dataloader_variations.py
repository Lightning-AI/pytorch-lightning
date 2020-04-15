from abc import ABC, abstractmethod


class TrainDataloaderVariationsMixin(ABC):

    @abstractmethod
    def dataloader(self, train):
        """placeholder"""
        pass

    def train_dataloader(self):
        return self.dataloader(
            train=True,
        )
