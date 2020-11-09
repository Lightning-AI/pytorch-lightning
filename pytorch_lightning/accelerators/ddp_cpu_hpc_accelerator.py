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
from pytorch_lightning.accelerators.ddp_hpc_accelerator import DDPHPCAccelerator


try:
    from hydra.utils import to_absolute_path, get_original_cwd
    from hydra.core.hydra_config import HydraConfig
except ImportError:
    HYDRA_AVAILABLE = False
else:
    HYDRA_AVAILABLE = True


class DDPCPUHPCAccelerator(DDPHPCAccelerator):

    def __init__(self, trainer, cluster_environment=None, ddp_plugin=None):
        """
        Runs training using DDP (with CPUs) strategy on a cluster

        Example::

            # default
            trainer = Trainer(accelerator=DDPCPUHPCAccelerator())

        """
        super().__init__(trainer, cluster_environment, ddp_plugin)
        self.nickname = 'ddp_cpu'

    def model_to_device(self, model, process_idx):
        model.cpu()

    def get_device_ids(self):
        device_ids = None
        return device_ids
