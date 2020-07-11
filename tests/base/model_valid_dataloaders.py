from abc import ABC, abstractmethod

from tests.base.dataloaders import CustomInfDataloader
from tests.base.dataloaders import CustomNotImplementedErrorDataloader
from tests.base import TrialMNIST
from torch.utils.data import DataLoader


class ValDataloaderVariations(ABC):

    @abstractmethod
    def dataloader(self, *args, **kwargs):
        """placeholder"""

    def val_dataloader(self):
        return self.dataloader(train=False)

    def val_dataloader__multiple_mixed_length(self):
        lengths = [100, 30]
        dataloaders = [self.dataloader(train=False, num_samples=n) for n in lengths]
        return dataloaders

    def val_dataloader__multiple(self):
        return [self.dataloader(train=False),
                self.dataloader(train=False)]

    def val_dataloader__long(self):
        dataset = DataLoader(TrialMNIST(download=True, train=False,
                                        num_samples=15000, digits=(0, 1, 2, 5, 8)), batch_size=32)
        return dataset

    def val_dataloader__infinite(self):
        return CustomInfDataloader(self.dataloader(train=False))

    def val_dataloader__not_implemented_error(self):
        return CustomNotImplementedErrorDataloader(self.dataloader(train=False))
