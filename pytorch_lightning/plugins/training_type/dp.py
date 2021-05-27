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
from typing import List, Optional

import torch
from torch.nn import DataParallel

from pytorch_lightning.overrides.data_parallel import LightningParallelModule
from pytorch_lightning.plugins.training_type.parallel import ParallelPlugin
from pytorch_lightning.trainer.connectors.logger_connector.result import Result
from pytorch_lightning.utilities.apply_func import apply_to_collection


class DataParallelPlugin(ParallelPlugin):
    """
    Implements data-parallel training in a single process, i.e., the model gets replicated to each
    device and each gets a split of the data.
    """

    def __init__(self, parallel_devices: Optional[List[torch.device]]):
        super().__init__(parallel_devices=parallel_devices, cluster_environment=None)

    @property
    def global_rank(self) -> int:
        return 0

    @property
    def local_rank(self) -> int:
        return 0

    @property
    def node_rank(self) -> int:
        return 0

    @property
    def world_size(self) -> int:
        return 1

    def setup(self, model):
        # model needs to be moved to the device before it is wrapped
        model.to(self.root_device)
        self._model = DataParallel(LightningParallelModule(model), self.parallel_devices)

    def reduce(self, tensor, *args, **kwargs):
        """
        Reduces a tensor from all parallel processes to one aggregated tensor.

        Args:
            tensor: the tensor to sync and reduce
            *args: ignored for DP
            **kwargs: ignored for DP

        Return:
            reduced value, except when the input was not a tensor the output remains is unchanged
        """
        if isinstance(tensor, Result):
            tensor.dp_reduce()

        else:

            def _reduce(t: torch.Tensor):
                dtype_tensor = t.dtype
                return t.float().mean().type(dtype_tensor)

            tensor = apply_to_collection(tensor, torch.Tensor, _reduce)

        return tensor

    @property
    def root_device(self):
        return self.parallel_devices[0]

    def model_to_device(self):
        # no need to do anything when model is wrapped in torch.nn.DataParallel
        pass

    def barrier(self, *args, **kwargs):
        pass

    def broadcast(self, obj: object, src: int = 0) -> object:
        return obj

    def reduce_boolean_decision(self, decision: bool) -> bool:
        return decision

    def training_step(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def validation_step(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def test_step(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def predict_step(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step_end(self, output):
        return self.reduce(output)

    def validation_step_end(self, output):
        return self.reduce(output)

    def test_step_end(self, output):
        return self.reduce(output)
