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

from pytorch_lightning import _logger as log
from pytorch_lightning.utilities import AMPType
from pytorch_lightning.utilities.distributed import rank_zero_only, find_free_network_port

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


class DDPBackend(object):

    def __init__(self, trainer):
        self.trainer = trainer
        self.task_idx = None
        self._has_spawned_children = False

    def slurm_setup(self):
        self.task_idx = int(os.environ['SLURM_LOCALID'])

    def torchelastic_setup(self):
        self.task_idx = int(os.environ['LOCAL_RANK'])

    def train(self, model):
        self.ddp_train(process_idx=self.task_idx, mp_queue=None, model=model)

    def spawn_ddp_children(self, model):
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

        local_rank = 0
        results = self.ddp_train(local_rank, mp_queue=None, model=model, is_master=True)
        del os.environ['WORLD_SIZE']

        return results

    def ddp_train(self, process_idx, mp_queue, model, is_master=False, proc_offset=0):
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
        if (self.trainer.node_rank != 0 or process_idx != 0) and self.trainer.progress_bar_callback is not None:
            self.trainer.progress_bar_callback.disable()

        # determine which process we are and world size
        self.trainer.local_rank = process_idx
        self.trainer.global_rank = self.trainer.node_rank * self.trainer.num_processes + process_idx
        self.trainer.world_size = self.trainer.num_nodes * self.trainer.num_processes

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

        # MODEL
        # copy model to each gpu
        if self.trainer.on_gpu:
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

        # CHOOSE OPTIMIZER
        # allow for lr schedulers as well
        optimizers, lr_schedulers, optimizer_frequencies = self.trainer.init_optimizers(model)
        self.trainer.optimizers = optimizers
        self.trainer.lr_schedulers = lr_schedulers
        self.trainer.optimizer_frequencies = optimizer_frequencies

        # set model properties before going into wrapper
        self.trainer.copy_trainer_model_properties(model)

        # AMP - run through amp wrapper before going to distributed DP
        if self.trainer.amp_backend == AMPType.APEX:
            model, optimizers = model.configure_apex(amp, model, self.trainer.optimizers, self.trainer.amp_level)
            self.trainer.optimizers = optimizers
            self.trainer.reinit_scheduler_properties(self.trainer.optimizers, self.trainer.lr_schedulers)

        # DDP2 uses all GPUs on the machine
        if self.trainer.distributed_backend == 'ddp' or self.trainer.distributed_backend == 'ddp_spawn':
            device_ids = [self.trainer.root_gpu]
        else:  # includes ddp_cpu
            device_ids = None

        # allow user to configure ddp
        model = model.configure_ddp(model, device_ids)

        # continue training routine
        results = self.trainer.run_pretrain_routine(model)

        # get original model
        model = self.trainer.get_model()

        # persist info in ddp_spawn
        self.trainer.transfer_distrib_spawn_state_on_fit_end(model, mp_queue, results)

        # clean up memory
        torch.cuda.empty_cache()

        if self.trainer.global_rank == 0 and self.trainer.distributed_backend not in ['ddp_spawn', 'ddp_cpu']:
            return results

    def _check_can_spawn_children(self):
        if self._has_spawned_children:
            raise RuntimeError(
                "You tried to run `.fit` or `.test` multiple times in the same script."
                " This is not supported in DDP mode, switch to `distributed_backend='ddp_spawn'` instead."
            )
