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

from pytorch_lightning.utilities.distributed import find_free_network_port
from pytorch_lightning.accelerators.ddp_base_backend import DDPBase

try:
    from apex import amp
except ImportError:
    amp = None


class DDPCPUSpawnBackend(DDPBase):

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
        mp.spawn(self.ddp_train_tmp, nprocs=self.nprocs, args=(self.mp_queue, model,))

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

    def set_world_ranks(self, process_idx):
        self.trainer.local_rank = process_idx
        self.trainer.global_rank = self.trainer.node_rank * self.trainer.num_processes + process_idx
        self.trainer.world_size = self.trainer.num_nodes * self.trainer.num_processes

    def model_to_device(self, model, process_idx, is_master):
        pass

    def get_device_ids(self):
        device_ids = None
        return device_ids
