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
import torch.multiprocessing as mp

from pytorch_lightning import _logger as log
from pytorch_lightning.accelerators.base import LightningBackend
from pytorch_lightning.utilities import AMPType
from pytorch_lightning.utilities.distributed import rank_zero_only, find_free_network_port

try:
    from apex import amp
except ImportError:
    amp = None


class DDPSpawnBackend(LightningBackend):

    def __init__(self, trainer):
        super().__init__(trainer)
        self.mp_queue = None

    def setup(self, model):
        super().setup(model)
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', str(find_free_network_port()))

        # pass in a state q
        smp = mp.get_context('spawn')
        self.mp_queue = smp.SimpleQueue()

    def train(self, nprocs):
        mp.spawn(self.ddp_train, nprocs=nprocs, args=(self.mp_queue, self._model,))

    def teardown(self):
        # restore main state with best weights
        best_path = self.mp_queue.get()
        results = self.mp_queue.get()
        last_path = self.mp_queue.get()

        # transfer back the best path to the trainer
        if self._trainer.checkpoint_callback:
            self._trainer.checkpoint_callback.best_model_path = best_path
        # todo, pass also bets score

        # load last weights
        if last_path is not None and not self._trainer.testing:
            ckpt = torch.load(last_path, map_location=lambda storage, loc: storage)
            self._model.load_state_dict(ckpt)

        self._trainer.model = self._model
        return results

    def ddp_train(self, process_idx, mp_queue, model):
        """
        Entry point for ddp

        Args:
            process_idx:
            mp_queue: multiprocessing queue
            model:

        Returns:

        """
        # show progressbar only on progress_rank 0
        if (self._trainer.node_rank != 0 or process_idx != 0) and self._trainer.progress_bar_callback is not None:
            self._trainer.progress_bar_callback.disable()

        # determine which process we are and world size
        if self._trainer.use_ddp:
            self._trainer.local_rank = process_idx
            self._trainer.global_rank = self._trainer.node_rank * self._trainer.num_processes + process_idx
            self._trainer.world_size = self._trainer.num_nodes * self._trainer.num_processes

        elif self._trainer.use_ddp2:
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

        # call sync_bn before .cuda(), configure_apex and configure_ddp
        if self._trainer.sync_batchnorm:
            model = model.configure_sync_batchnorm(model)

        # MODEL
        # copy model to each gpu
        if self._trainer.on_gpu:
            gpu_idx = process_idx
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

        # AMP -
        # run through amp wrapper before going to distributed DP
        if self._trainer.amp_backend == AMPType.APEX:
            model, optimizers = model.configure_apex(amp, model, self._trainer.optimizers, self._trainer.amp_level)
            self._trainer.optimizers = optimizers
            self._trainer.reinit_scheduler_properties(self._trainer.optimizers, self._trainer.lr_schedulers)

        # DDP2 uses all GPUs on the machine
        if self._trainer.distributed_backend == 'ddp' or self._trainer.distributed_backend == 'ddp_spawn':
            device_ids = [self._trainer.root_gpu]
        elif self._trainer.use_ddp2:
            device_ids = self._trainer.data_parallel_device_ids
        else:  # includes ddp_cpu
            device_ids = None

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
