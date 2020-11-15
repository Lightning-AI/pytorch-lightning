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

import torch.multiprocessing as mp

from pytorch_lightning.accelerators.distributed_accelerator import DistributedAccelerator
from pytorch_lightning.distributed.dist import LightningDistributed
from pytorch_lightning.utilities.distributed import find_free_network_port
from pytorch_lightning.utilities.distributed import rank_zero_warn

try:
    from hydra.core.hydra_config import HydraConfig
    from hydra.utils import get_original_cwd, to_absolute_path
except ImportError:
    HYDRA_AVAILABLE = False
else:
    HYDRA_AVAILABLE = True


class DDPCPUSpawnAccelerator(DistributedAccelerator):

    def __init__(self, trainer, nprocs, cluster_environment=None, ddp_plugin=None):
        """
        Runs training using DDP (on a single machine or manually on multiple machines), using mp.spawn

        Example::

            # default
            trainer = Trainer(accelerator=DDPCPUSpawnAccelerator())

        """
        super().__init__(trainer, cluster_environment, ddp_plugin)
        self.mp_queue = None
        self.nprocs = nprocs
        self.dist = LightningDistributed()
        self.nickname = 'ddp_cpu'

    def setup(self, model):
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', str(find_free_network_port()))

        # pass in a state q
        smp = mp.get_context('spawn')
        self.mp_queue = smp.SimpleQueue()

        self.trainer.model = model

    def train(self):
        model = self.trainer.model

        # train in children process
        mp.spawn(self.ddp_train_spawn, nprocs=self.nprocs, args=(self.mp_queue, model,))

        # restore main state with best weights
        best_path = self.mp_queue.get()
        results = self.mp_queue.get()

        # recover the weights of the processes trained in the children
        self.__recover_child_process_weights(model, best_path)
        return results

    def ddp_train_spawn(self, process_idx, mp_queue, model):
        """
        Entry point for ddp spawn. Ensures we transfer spawn state to main process before fit end.

        Args:
            process_idx: Rank of the process.
            mp_queue: Multiprocessing queue.
            model: Model that has been trained in trainer fit.
        """
        results = self.ddp_train(process_idx, model)

        # persist info in ddp_spawn
        self.transfer_distrib_spawn_state_on_fit_end(model, mp_queue, results)

    def model_to_device(self, model, process_idx):
        model.cpu()

    def get_device_ids(self):
        device_ids = None
        return device_ids

    def __recover_child_process_weights(self, model, best_path):
        # transfer back the best path to the trainer
        if self.trainer.checkpoint_callback:
            self.trainer.checkpoint_callback.best_model_path = best_path

        self.trainer.model = model

    def transfer_distrib_spawn_state_on_fit_end(self, model, mp_queue, results):
        # track the best model path
        best_model_path = None
        if self.trainer.checkpoint_callback is not None:
            best_model_path = self.trainer.checkpoint_callback.best_model_path

        if self.trainer.global_rank == 0 and mp_queue is not None:
            rank_zero_warn('cleaning up ddp environment...')
            # todo, pass complete checkpoint as state dictionary
            mp_queue.put(best_model_path)
            mp_queue.put(results)
