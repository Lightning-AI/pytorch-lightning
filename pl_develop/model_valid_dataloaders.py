from abc import ABC, abstractmethod

from pl_develop import CustomInfDataloader
from pl_develop import CustomNotImplementedErrorDataloader


class ValDataloaderVariations(ABC):

    @abstractmethod
    def dataloader(self, train: bool):
        """placeholder"""

    def val_dataloader(self):
        return self.dataloader(train=False)

    def val_dataloader__multiple(self):
        return [self.dataloader(train=False),
                self.dataloader(train=False)]

    def val_dataloader__infinite(self):
        return CustomInfDataloader(self.dataloader(train=False))

    def val_dataloader__not_implemented_error(self):
        return CustomNotImplementedErrorDataloader(self.dataloader(train=False))
