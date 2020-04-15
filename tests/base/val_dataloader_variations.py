from abc import ABC, abstractmethod


class ValDataloaderVariationsMixin(ABC):

    @abstractmethod
    def dataloader(self, train):
        """placeholder"""
        pass

    def val_dataloader(self):
        return self.dataloader(
            train=False,
        )
