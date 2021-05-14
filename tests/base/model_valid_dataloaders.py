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
        return [
            self.dataloader(train=False),
            self.dataloader(train=False),
        ]

    def val_dataloader__infinite(self):
        return CustomInfDataloader(self.dataloader(train=False))

    def val_dataloader__not_implemented_error(self):
        return CustomNotImplementedErrorDataloader(self.dataloader(train=False))
