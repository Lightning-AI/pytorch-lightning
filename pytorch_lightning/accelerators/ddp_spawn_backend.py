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
import re

import torch

from pytorch_lightning import _logger as log
from pytorch_lightning.accelerators.base_backend import Accelerator
from pytorch_lightning.utilities.cloud_io import atomic_save
from pytorch_lightning.utilities.distributed import rank_zero_only, rank_zero_warn
from pytorch_lightning.utilities.distributed import find_free_network_port


class DDPSpawnBackend(Accelerator):

    def __init__(self, trainer, nprocs):
        super().__init__(trainer)
        self.mp_queue = None
        self.nprocs = nprocs

    def setup(self, model):
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', str(find_free_network_port()))

        # pass in a state q
        smp = mp.get_context('spawn')
        self.mp_queue = smp.SimpleQueue()

        self.trainer.model = model

    def train(self):
        model = self.trainer.model

        # train in children process
        mp.spawn(self.ddp_train, nprocs=self.nprocs, args=(self.mp_queue, model,))

        # restore main state with best weights
        best_path = self.mp_queue.get()
        results = self.mp_queue.get()
        last_path = self.mp_queue.get()

        # recover the weights of the processes trained in the children
        self.__recover_child_process_weights(model, best_path, last_path)
        return results

    def __recover_child_process_weights(self, model, best_path, last_path):
        # transfer back the best path to the trainer
        if self.trainer.checkpoint_callback:
            self.trainer.checkpoint_callback.best_model_path = best_path
        # todo, pass also best score

        # load last weights
        if last_path is not None and not self.trainer.testing:
            ckpt = torch.load(last_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(ckpt)

        self.trainer.model = model

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
        if (self.trainer.node_rank != 0 or process_idx != 0) and self.trainer.progress_bar_callback is not None:
            self.trainer.progress_bar_callback.disable()

        # determine which process we are and world size
        self.set_world_ranks(process_idx)

        # set warning rank
        rank_zero_only.rank = self.trainer.global_rank

        # set up server using proc 0's ip address
        # try to init for 20 times at max in case ports are taken
        # where to store ip_table
        model.trainer = self.trainer
        model.init_ddp_connection(
            self.trainer.global_rank,
            self.trainer.world_size,
            self.trainer.is_slurm_managing_tasks
        )

        # call setup after the ddp process has connected
        self.trainer.call_setup_hook(model)

        # on world_size=0 let everyone know training is starting
        if self.trainer.is_global_zero:
            log.info('-' * 100)
            log.info(f'distributed_backend={self.trainer.distributed_backend}')
            log.info(f'All DDP processes registered. Starting ddp with {self.trainer.world_size} processes')
            log.info('-' * 100)

        # call sync_bn before .cuda(), configure_apex and configure_ddp
        if self.trainer.sync_batchnorm:
            model = model.configure_sync_batchnorm(model)

        # move the model to the correct device
        self.model_to_device(model, process_idx)

        # CHOOSE OPTIMIZER
        # allow for lr schedulers as well
        self.setup_optimizers(model)

        # set model properties before going into wrapper
        self.trainer.model_connector.copy_trainer_model_properties(model)

        # 16-bit
        model = self.trainer.precision_connector.connect(model)

        # device ids change depending on the DDP setup
        device_ids = self.get_device_ids()

        # allow user to configure ddp
        model = model.configure_ddp(model, device_ids)

        # set up training routine
        self.trainer.train_loop.setup_training(model)

        # train or test
        results = self.train_or_test()

        # get original model
        model = self.trainer.get_model()

        # persist info in ddp_spawn
        self.transfer_distrib_spawn_state_on_fit_end(model, mp_queue, results)

    def set_world_ranks(self, process_idx):
        self.trainer.local_rank = process_idx
        self.trainer.global_rank = self.trainer.node_rank * self.trainer.num_processes + process_idx
        self.trainer.world_size = self.trainer.num_nodes * self.trainer.num_processes

    def model_to_device(self, model, process_idx):
        gpu_idx = process_idx
        self.trainer.root_gpu = gpu_idx
        torch.cuda.set_device(self.trainer.root_gpu)
        model.cuda(self.trainer.root_gpu)

    def get_device_ids(self):
        device_ids = [self.trainer.root_gpu]
        return device_ids

    def transfer_distrib_spawn_state_on_fit_end(self, model, mp_queue, results):
        best_model_path = None
        if self.trainer.checkpoint_callback is not None:
            best_model_path = self.trainer.checkpoint_callback.best_model_path

        if self.trainer.global_rank == 0 and mp_queue is not None:
            rank_zero_warn('cleaning up ddp environment...')
            # todo, pass complete checkpoint as state dictionary
            mp_queue.put(best_model_path)
            mp_queue.put(results)

            # save the last weights
            last_path = None
            if not self.trainer.testing and best_model_path is not None and len(best_model_path) > 0:
                last_path = re.sub('.ckpt', '.tmp_end.ckpt', best_model_path)
                atomic_save(model.state_dict(), last_path)
            mp_queue.put(last_path)
