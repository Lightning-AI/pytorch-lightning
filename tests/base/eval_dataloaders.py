from abc import ABC, abstractmethod


class EvalDataloaderVariations(ABC):

    @abstractmethod
    def dataloader(self, train):
        """placeholder"""

    def test_dataloader(self):
        return self.dataloader(
            train=False,
        )
