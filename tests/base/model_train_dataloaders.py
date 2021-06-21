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
from abc import ABC, abstractmethod

from tests.helpers.dataloaders import CustomInfDataloader, CustomNotImplementedErrorDataloader


class TrainDataloaderVariations(ABC):

    @abstractmethod
    def dataloader(self, train: bool, *args, **kwargs):
        """placeholder"""

    def train_dataloader(self):
        return self.dataloader(train=True)

    def train_dataloader__infinite(self):
        return CustomInfDataloader(self.dataloader(train=True))

    def train_dataloader__not_implemented_error(self):
        return CustomNotImplementedErrorDataloader(self.dataloader(train=True))

    def train_dataloader__zero_length(self):
        dataloader = self.dataloader(train=True)
        dataloader.dataset.data = dataloader.dataset.data[:0]
        dataloader.dataset.targets = dataloader.dataset.targets[:0]
        return dataloader

    def train_dataloader__multiple_mapping(self):
        """Return a mapping loaders with different lengths"""

        # List[DataLoader]
        loaders_a_b = [self.dataloader(num_samples=100, train=True), self.dataloader(num_samples=50, train=True)]
        loaders_c_d_e = [
            self.dataloader(num_samples=50, train=True),
            self.dataloader(num_samples=50, train=True),
            self.dataloader(num_samples=50, train=True)
        ]
        # Dict[str, List[DataLoader]]
        loaders = {"a_b": loaders_a_b, "c_d_e": loaders_c_d_e}
        return loaders
