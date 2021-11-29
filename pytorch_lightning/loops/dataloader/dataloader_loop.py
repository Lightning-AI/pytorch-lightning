# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import abstractmethod
from typing import Any, Sequence

from torch.utils.data import DataLoader

from pytorch_lightning.loops.base import Loop
from pytorch_lightning.trainer.progress import DataLoaderProgress


class DataLoaderLoop(Loop):
    """Base class to loop over all dataloaders."""

    def __init__(self) -> None:
        super().__init__()
        self.dataloader_progress = DataLoaderProgress()

    @property
    @abstractmethod
    def dataloaders(self) -> Sequence[DataLoader]:
        """Returns the dataloaders to loop over."""

    @property
    def current_dataloader_idx(self) -> int:
        """Returns the index of the current dataloader."""
        return self.dataloader_progress.current.ready - 1

    @property
    def current_dataloader(self) -> DataLoader:
        """Returns the current dataloader."""
        return self.dataloaders[self.current_dataloader_idx]

    @property
    def num_dataloaders(self) -> int:
        """Returns the number of dataloaders present."""
        return len(self.dataloaders) if self.dataloaders is not None else 0

    @property
    def done(self) -> bool:
        """Returns whether all dataloaders have been processed."""
        return self.dataloader_progress.current.completed >= self.num_dataloaders

    def reset(self) -> None:
        """Resets the internal state."""
        if not self.restarting:
            self.dataloader_progress.reset_on_run()
        else:
            self.dataloader_progress.reset_on_restart()

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        self.dataloader_progress.increment_ready()

    def on_advance_end(self) -> None:
        self.dataloader_progress.increment_completed()
