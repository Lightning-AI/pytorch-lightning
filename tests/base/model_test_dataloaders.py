from abc import ABC, abstractmethod

from tests.base.dataloaders import CustomInfDataloader
from torch.utils.data import DataLoader
from tests.base import TrialMNIST
from tests.base.dataloaders import CustomNotImplementedErrorDataloader


class TestDataloaderVariations(ABC):

    @abstractmethod
    def dataloader(self, *args, **kwargs):
        """placeholder"""

    def test_dataloader(self):
        return self.dataloader(train=False)

    def test_dataloader__infinite(self):
        return CustomInfDataloader(self.dataloader(train=False))

    def test_dataloader__long(self):
        dataset = DataLoader(TrialMNIST(download=True, train=False, num_samples=15000, digits=(0, 1, 2, 5, 8)),
                             batch_size=32)
        return dataset

    def test_dataloader__not_implemented_error(self):
        return CustomNotImplementedErrorDataloader(self.dataloader(train=False))

    def test_dataloader__multiple_mixed_length(self):
        lengths = [50, 30, 40]
        dataloaders = [self.dataloader(train=False, num_samples=n) for n in lengths]
        return dataloaders

    def test_dataloader__empty(self):
        return None

    def test_dataloader__multiple(self):
        return [self.dataloader(train=False), self.dataloader(train=False)]
