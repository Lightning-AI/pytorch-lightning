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
from pytorch_lightning.accelerators.base import LightningBackend
from pytorch_lightning.utilities import AMPType
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities.exceptions import MisconfigurationException

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


class DDP2Backend(LightningBackend):

    def __init__(self, trainer):
        super().__init__(trainer)
        self.task_idx = None

    def setup(self, model):
        super().setup(model)
        self._resolve_task_idx()

    def _resolve_task_idx(self):
        if self._trainer.is_slurm_managing_tasks:
            self.task_idx = int(os.environ['SLURM_LOCALID'])
        else:
            # torchelastic or general non_slurm ddp2
            try:
                self.task_idx = int(os.environ['LOCAL_RANK'])
            except Exception as e:
                m = 'ddp2 only works in SLURM or via torchelastic with the WORLD_SIZE, LOCAL_RANK, GROUP_RANK flags'
                raise MisconfigurationException(m)

    def train(self):
        self.ddp_training(process_idx=self.task_idx, mp_queue=None, model=self._model)

    def ddp_training(self, process_idx, mp_queue, model, is_master=False, proc_offset=0):
        """
        Entry point for ddp

        Args:
            process_idx:
            mp_queue: multiprocessing queue
            model:
            is_master:
            proc_offset:

        Returns:

        """
        # offset the process id if requested
        process_idx = process_idx + proc_offset

        # show progressbar only on progress_rank 0
        if (self._trainer.node_rank != 0 or process_idx != 0) and self._trainer.progress_bar_callback is not None:
            self._trainer.progress_bar_callback.disable()

        self._trainer.local_rank = self._trainer.node_rank
        self._trainer.global_rank = self._trainer.node_rank
        self._trainer.world_size = self._trainer.num_nodes

        # set warning rank
        rank_zero_only.rank = self._trainer.global_rank

        # set up server using proc 0's ip address
        # try to init for 20 times at max in case ports are taken
        # where to store ip_table
        model.trainer = self._trainer
        model.init_ddp_connection(
            self._trainer.global_rank,
            self._trainer.world_size,
            self._trainer.is_slurm_managing_tasks
        )

        # call setup after the ddp process has connected
        self._trainer.call_setup_hook(model)

        # on world_size=0 let everyone know training is starting
        if self._trainer.is_global_zero:
            log.info('-' * 100)
            log.info(f'distributed_backend={self._trainer.distributed_backend}')
            log.info(f'All DDP processes registered. Starting ddp with {self._trainer.world_size} processes')
            log.info('-' * 100)

        # MODEL
        # copy model to each gpu
        if self._trainer.on_gpu:
            gpu_idx = process_idx

            # when using ddp, the master process (proc 0) continues running as the main one
            # this means that the local rank will always be 0
            # (even if cuda visible devices has other visible gpus)
            # this means that the master process needs to pull the 0th visible index as the device number
            if is_master:
                available_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
                gpu_idx = int(available_gpus[self._trainer.local_rank])

            self._trainer.root_gpu = gpu_idx
            torch.cuda.set_device(self._trainer.root_gpu)
            model.cuda(self._trainer.root_gpu)

        # CHOOSE OPTIMIZER
        # allow for lr schedulers as well
        optimizers, lr_schedulers, optimizer_frequencies = self._trainer.init_optimizers(model)
        self._trainer.optimizers = optimizers
        self._trainer.lr_schedulers = lr_schedulers
        self._trainer.optimizer_frequencies = optimizer_frequencies

        # set model properties before going into wrapper
        self._trainer.copy_trainer_model_properties(model)

        # AMP - run through amp wrapper before going to distributed DP
        if self._trainer.amp_backend == AMPType.APEX:
            model, optimizers = model.configure_apex(amp, model, self._trainer.optimizers, self._trainer.amp_level)
            self._trainer.optimizers = optimizers
            self._trainer.reinit_scheduler_properties(self._trainer.optimizers, self._trainer.lr_schedulers)

        # DDP2 uses all GPUs on the machine
        device_ids = self._trainer.data_parallel_device_ids

        # allow user to configure ddp
        model = model.configure_ddp(model, device_ids)

        # continue training routine
        results = self._trainer.run_pretrain_routine(model)

        # get original model
        model = self._trainer.get_model()

        # persist info in ddp_spawn
        self._trainer.transfer_distrib_spawn_state_on_fit_end(model, mp_queue, results)

        # clean up memory
        torch.cuda.empty_cache()
