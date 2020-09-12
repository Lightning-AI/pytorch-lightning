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

import os

import torch

from pytorch_lightning import _logger as log
from pytorch_lightning.utilities import AMPType
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.core.step_result import Result
from pytorch_lightning.accelerators.ddp_base_backend import DDPBase

try:
    from hydra.utils import to_absolute_path, get_original_cwd
    from hydra.core.hydra_config import HydraConfig
except ImportError:
    HYDRA_AVAILABLE = False
else:
    HYDRA_AVAILABLE = True

try:
    from apex import amp
except ImportError:
    amp = None


class DDP2Backend(DDPBase):

    def __init__(self, trainer):
        super().__init__(trainer)
        self.task_idx = None

    def setup(self, model):
        self._resolve_task_idx()

        self.trainer.model = model

    def _resolve_task_idx(self):
        if self.trainer.is_slurm_managing_tasks:
            self.task_idx = int(os.environ['SLURM_LOCALID'])
        else:
            # torchelastic or general non_slurm ddp2
            try:
                self.task_idx = int(os.environ['LOCAL_RANK'])
            except Exception as e:
                m = 'ddp2 only works in SLURM or via torchelastic with the WORLD_SIZE, LOCAL_RANK, GROUP_RANK flags'
                raise MisconfigurationException(m)

    def train(self):
        model = self.trainer.model
        self.ddp_train_tmp(process_idx=self.task_idx, mp_queue=None, model=model)

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

    def model_to_device(self, model, process_idx, is_master):
        gpu_idx = process_idx

        # when using ddp, the master process (proc 0) continues running as the main one
        # this means that the local rank will always be 0
        # (even if cuda visible devices has other visible gpus)
        # this means that the master process needs to pull the 0th visible index as the device number
        if is_master:
            available_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
            gpu_idx = int(available_gpus[self.trainer.local_rank])

        self.trainer.root_gpu = gpu_idx
        torch.cuda.set_device(self.trainer.root_gpu)
        model.cuda(self.trainer.root_gpu)

    def get_device_ids(self):
        device_ids = self.trainer.data_parallel_device_ids
        return device_ids
