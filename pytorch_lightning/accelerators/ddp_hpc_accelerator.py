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

from pytorch_lightning.accelerators.distributed_accelerator import DistributedAccelerator
from pytorch_lightning.distributed.dist import LightningDistributed

try:
    from hydra.utils import to_absolute_path, get_original_cwd
    from hydra.core.hydra_config import HydraConfig
except ImportError:
    HYDRA_AVAILABLE = False
else:
    HYDRA_AVAILABLE = True


class DDPHPCAccelerator(DistributedAccelerator):

    def __init__(self, trainer, cluster_environment=None, ddp_plugin=None):
        """
        Runs training using DDP on an HPC cluster

        Example::

            # default
            trainer = Trainer(accelerator=DDPHPCAccelerator())

        """
        super().__init__(trainer, cluster_environment, ddp_plugin)
        self.task_idx = None
        self._has_spawned_children = False
        self.dist = LightningDistributed()
        self.nickname = 'ddp'

    def setup(self, model):
        self.trainer.model = model
        self.task_idx = self.cluster_environment.local_rank()

    def train(self):
        model = self.trainer.model
        self.ddp_train(process_idx=self.task_idx, model=model)
