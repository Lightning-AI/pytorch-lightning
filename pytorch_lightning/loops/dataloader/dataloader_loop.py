from abc import abstractmethod
from typing import Sequence

from torch.utils.data import DataLoader

from pytorch_lightning.loops.base import Loop


# TODO: Handle max_batches also in base class here
class DataLoaderLoop(Loop):

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def dataloaders(self) -> Sequence[DataLoader]:
        pass

    @property
    def current_dataloader_idx(self) -> int:
        return self.iteration_count

    @property
    def current_dataloader(self) -> DataLoader:
        return self.dataloaders[self.current_dataloader_idx]

    @property
    def num_dataloaders(self) -> int:
        return len(self.dataloaders)

    @property
    def done(self) -> bool:
        return self.current_dataloader_idx >= self.num_dataloaders

    def reset(self) -> None:
        self.iteration_count = 0
