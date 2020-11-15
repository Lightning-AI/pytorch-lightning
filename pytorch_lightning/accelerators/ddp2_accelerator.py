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
# limitations under the License

import torch

from pytorch_lightning.accelerators.distributed_accelerator import DistributedAccelerator
from pytorch_lightning.core.step_result import Result
from pytorch_lightning.distributed.dist import LightningDistributed

try:
    from hydra.utils import to_absolute_path, get_original_cwd
    from hydra.core.hydra_config import HydraConfig
except ImportError:
    HYDRA_AVAILABLE = False
else:
    HYDRA_AVAILABLE = True


class DDP2Accelerator(DistributedAccelerator):

    def __init__(self, trainer, cluster_environment=None, ddp_plugin=None):
        """
        Runs training using DDP2 strategy on a cluster

        Example::

            # default
            trainer = Trainer(accelerator=DDP2Accelerator())

        """
        super().__init__(trainer, cluster_environment, ddp_plugin)
        self.task_idx = None
        self.dist = LightningDistributed()
        self.nickname = 'ddp2'

    def setup(self, model):
        self.trainer.model = model
        self.task_idx = self.cluster_environment.local_rank()

    def train(self):
        model = self.trainer.model
        return self.ddp_train(process_idx=self.task_idx, model=model)

    def training_step_end(self, output):
        if isinstance(output, Result):
            output.dp_reduce()
        return output

    def validation_step_end(self, output):
        if isinstance(output, Result):
            output.dp_reduce()
        return output

    def test_step_end(self, output):
        if isinstance(output, Result):
            output.dp_reduce()
        return output

    def set_world_ranks(self, process_idx):
        self.trainer.local_rank = self.trainer.node_rank
        self.trainer.global_rank = self.trainer.node_rank
        self.trainer.world_size = self.trainer.num_nodes

    def model_to_device(self, model, process_idx):
        self.trainer.root_gpu = process_idx
        torch.cuda.set_device(self.trainer.root_gpu)
        model.cuda(self.trainer.root_gpu)

    def get_device_ids(self):
        device_ids = self.trainer.data_parallel_device_ids
        return device_ids
