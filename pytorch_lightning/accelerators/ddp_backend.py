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
import subprocess
import sys
from os.path import abspath
from time import sleep
from typing import Optional

import numpy as np
import torch

from pytorch_lightning.utilities.distributed import find_free_network_port
from pytorch_lightning.accelerators.ddp_base_backend import DDPBase

try:
    from hydra.utils import to_absolute_path, get_original_cwd
    from hydra.core.hydra_config import HydraConfig
except ImportError:
    HYDRA_AVAILABLE = False
else:
    HYDRA_AVAILABLE = True


class DDPBackend(DDPBase):

    def __init__(self, trainer, mode: str = 'ddp'):
        super().__init__(trainer)
        self.task_idx = None
        self._has_spawned_children = False
        self.mode = mode

    def setup(self, model):
        if self.mode == 'ddp':
            self.__ddp_script_mode_setup()
        elif self.mode == 'slurm_ddp':
            self.__slurm_setup()
        elif self.mode == 'torchelastic_ddp':
            self.__torchelastic_setup()

        self.trainer.model = model

    def __slurm_setup(self):
        self.task_idx = int(os.environ['SLURM_LOCALID'])

    def __torchelastic_setup(self):
        self.task_idx = int(os.environ['LOCAL_RANK'])

    def __ddp_script_mode_setup(self):
        assert self.trainer.global_rank == 0
        self._check_can_spawn_children()
        self._has_spawned_children = True

        os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', '127.0.0.1')
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', str(find_free_network_port()))

        # allow the user to pass the node rank
        node_rank = '0'
        node_rank = os.environ.get('NODE_RANK', node_rank)
        node_rank = os.environ.get('GROUP_RANK', node_rank)
        os.environ['NODE_RANK'] = node_rank
        os.environ['LOCAL_RANK'] = '0'

        # when user is using hydra find the absolute path
        path_lib = abspath if not HYDRA_AVAILABLE else to_absolute_path

        # pull out the commands used to run the script and resolve the abs file path
        command = sys.argv
        try:
            full_path = path_lib(command[0])
        except Exception as e:
            full_path = abspath(command[0])

        command[0] = full_path
        # use the same python interpreter and actually running
        command = [sys.executable] + command

        # the visible devices tell us how many GPUs we want to use.
        # when the trainer script was called the device has already been scoped by the time
        # code reaches this point. so, to call the scripts, we need to leave cuda visible devices alone
        # but forward the GPUs selected via environment variables
        gpu_ids = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if len(gpu_ids) == 1:
            gpu_ids = f'{gpu_ids},'

        num_gpus = max(1, len(gpu_ids.split(',')))

        # set the flag for ddp scripts
        os.environ['PL_TRAINER_GPUS'] = gpu_ids

        os.environ['WORLD_SIZE'] = f'{num_gpus * self.trainer.num_nodes}'

        self.trainer.interactive_ddp_procs = []
        for local_rank in range(1, self.trainer.num_processes):
            env_copy = os.environ.copy()
            env_copy['LOCAL_RANK'] = f'{local_rank}'

            # start process
            # if hydra is available and initialized, make sure to set the cwd correctly
            cwd: Optional[str] = None
            if HYDRA_AVAILABLE:
                if HydraConfig.initialized():
                    cwd = get_original_cwd()
            proc = subprocess.Popen(command, env=env_copy, cwd=cwd)
            self.trainer.interactive_ddp_procs.append(proc)

            # starting all processes at once can cause issues
            # with dataloaders delay between 1-10 seconds
            delay = np.random.uniform(1, 5, 1)[0]
            sleep(delay)

        self.task_idx = 0

    def train(self):
        model = self.trainer.model
        if self.mode == 'ddp':
            results = self.ddp_train_tmp(process_idx=self.task_idx, mp_queue=None, model=model, is_master=True)
            del os.environ['WORLD_SIZE']
            return results
        else:
            self.ddp_train_tmp(process_idx=self.task_idx, mp_queue=None, model=model)

    def _check_can_spawn_children(self):
        if self._has_spawned_children:
            raise RuntimeError(
                "You tried to run `.fit` or `.test` multiple times in the same script."
                " This is not supported in DDP mode, switch to `distributed_backend='ddp_spawn'` instead."
            )


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
        device_ids = [self.trainer.root_gpu]
        return device_ids
