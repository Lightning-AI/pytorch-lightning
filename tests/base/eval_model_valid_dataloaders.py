from abc import ABC, abstractmethod


class ValDataloaderVariations(ABC):

    @abstractmethod
    def dataloader(self, train: bool):
        """placeholder"""

    def val_dataloader(self):
        return self.dataloader(train=False)
