# Copyright The Lightning AI team.
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
"""Custom dataloaders for testing."""

import torch


class CustomInfDataloader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iter = iter(dataloader)
        self.count = 0
        self.dataloader.num_workers = 0  # reduce chance for hanging pytest

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count >= 50:
            raise StopIteration
        self.count = self.count + 1
        try:
            return next(self.iter)
        except StopIteration:
            self.iter = iter(self.dataloader)
            return next(self.iter)


class CustomNotImplementedErrorDataloader(CustomInfDataloader):
    def __len__(self):
        """Raise NotImplementedError."""
        raise NotImplementedError

    def __next__(self):
        if self.count >= 2:
            raise StopIteration
        self.count = self.count + 1
        try:
            return next(self.iter)
        except StopIteration:
            self.iter = iter(self.dataloader)
            return next(self.iter)


class ResumableIterable:
    """Yields a fixed sequence of batches and can resume from the middle of an epoch."""

    def __init__(self, num_batches=4):
        self.batches = torch.randn(num_batches, 1, 32, generator=torch.Generator().manual_seed(0))
        self.start = self.index = 0

    def __iter__(self):
        start, self.start = self.start, 0
        for i in range(start, len(self)):
            self.index = i
            yield self.batches[i]

    def __len__(self):
        return len(self.batches)

    def state_dict(self):
        return {"index": self.index}

    def load_state_dict(self, state_dict):
        self.start = state_dict["index"] + 1
