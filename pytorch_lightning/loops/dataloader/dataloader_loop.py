from abc import abstractmethod
from typing import Sequence

from torch.utils.data import DataLoader

from pytorch_lightning.loops.base import Loop


# TODO: Handle max_batches also in base class here
class DataLoaderLoop(Loop):
    """Base class to loop over all dataloaders"""

    @property
    @abstractmethod
    def dataloaders(self) -> Sequence[DataLoader]:
        """Returns the dataloaders to loop over."""

    @property
    def current_dataloader_idx(self) -> int:
        """Returns the index of the current dataloader"""
        return self.iteration_count

    @property
    def current_dataloader(self) -> DataLoader:
        """Returns the current datalaoder"""
        return self.dataloaders[self.current_dataloader_idx]

    @property
    def num_dataloaders(self) -> int:
        """Returns the number of dataloaders present"""
        return len(self.dataloaders)

    @property
    def done(self) -> bool:
        """Returns whether all dataloaders have been processed"""
        return self.current_dataloader_idx >= self.num_dataloaders

    def reset(self) -> None:
        """Resets the internal state"""
        self.iteration_count = 0
