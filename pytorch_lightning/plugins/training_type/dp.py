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
from pytorch_lightning.plugins.collective.collective_plugin import Collective
from pytorch_lightning.plugins.collective.torch_collective import TorchCollective
from pytorch_lightning.plugins.io.checkpoint_plugin import CheckpointIO
from pytorch_lightning.plugins.training_type.parallel import ParallelPlugin
from pytorch_lightning.utilities.model_helpers import is_overridden


class DataParallelPlugin(ParallelPlugin):
    """Implements data-parallel training in a single process, i.e., the model gets replicated to each device and
    each gets a split of the data."""

    def __init__(
        self,
        parallel_devices: Optional[List[torch.device]],
        checkpoint_io: Optional[CheckpointIO] = None,
        collective: Optional[Collective] = None,
    ):
        super().__init__(parallel_devices=parallel_devices, cluster_environment=None, checkpoint_io=checkpoint_io)
        super().__init__(
            parallel_devices=parallel_devices,
            cluster_environment=None,
            checkpoint_io=checkpoint_io,
            collective=collective or TorchCollective(local_reduce=True),
        )

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

    def setup(self) -> None:
        # model needs to be moved to the device before it is wrapped
        self.model_to_device()
        self._model = DataParallel(LightningParallelModule(self._model), self.parallel_devices)

    @property
    def root_device(self):
        return self.parallel_devices[0]

    def model_to_device(self) -> None:
        self._model.to(self.root_device)

    def training_step(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def validation_step(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def test_step(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def predict_step(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step_end(self, output):
        if not is_overridden("training_step_end", self.lightning_module):
            return self.reduce(output)
        return output

    def validation_step_end(self, output):
        if not is_overridden("validation_step_end", self.lightning_module):
            return self.reduce(output)
        return output

    def test_step_end(self, output):
        if not is_overridden("test_step_end", self.lightning_module):
            return self.reduce(output)
        return output

    def teardown(self) -> None:
        if self.on_gpu:
            # GPU teardown
            self.lightning_module.cpu()
            # clean up memory
            torch.cuda.empty_cache()
