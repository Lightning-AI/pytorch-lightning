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
from torch.utils.data import DataLoader

from tests.base.datasets import TrialMNIST


class ModelTemplateData:

    def dataloader(self, train: bool, num_samples: int = 100):
        dataset = TrialMNIST(root=self.data_root, train=train, num_samples=num_samples, download=True)

        loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=train,
        )
        return loader


class ModelTemplateUtils:

    def get_output_metric(self, output, name):
        if isinstance(output, dict):
            val = output[name]
        else:  # if it is 2level deep -> per dataloader and per batch
            val = sum(out[name] for out in output) / len(output)
        return val
