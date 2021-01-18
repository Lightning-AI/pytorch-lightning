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
from os.path import abspath
import subprocess
import sys
from time import sleep
from typing import Any, List, Optional, Union

import numpy as np
import torch
import torch.distributed as torch_distrib
from torch.nn.parallel import DistributedDataParallel

from pytorch_lightning import _logger as log
from pytorch_lightning.accelerators.accelerator import Accelerator, ReduceOp
from pytorch_lightning.cluster_environments import ClusterEnvironment
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.distributed.dist import LightningDistributed
from pytorch_lightning.plugins.ddp_plugin import DDPPlugin
from pytorch_lightning.plugins.rpc_plugin import RPCPlugin
from pytorch_lightning.utilities import AMPType, HYDRA_AVAILABLE
from pytorch_lightning.utilities.distributed import (
    all_gather_ddp_if_available,
    find_free_network_port,
    rank_zero_only,
    sync_ddp_if_available,
)
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.seed import seed_everything

if HYDRA_AVAILABLE:
    from hydra.core.hydra_config import HydraConfig
    from hydra.utils import get_original_cwd, to_absolute_path


class DDPAccelerator(Accelerator):

    def __init__(self,
                 trainer: Optional = None,
                 cluster_environment: Optional[ClusterEnvironment] = None,
                 ddp_plugin: Optional[DDPPlugin] = None):
        """
        Runs training using DDP strategy on a single machine (manually, not via cluster start)

        Example::

            # default
            trainer = Trainer(accelerator=DDPAccelerator())

        """
        super().__init__(trainer, cluster_environment, ddp_plugin)
        self.task_idx = None
        self._has_spawned_children = False
        self.interactive_ddp_procs = []
        self.dist = LightningDistributed()
        self.nickname = 'ddp'

    def setup(self, model):
        # first track model
        self.trainer.model = model

        # start the other scripts
        if os.environ.get('PL_IN_DDP_SUBPROCESS', '0') != '1':
            self._call_children_scripts()

        # set the task idx
        self.task_idx = int(os.environ['LOCAL_RANK'])

    def _call_children_scripts(self):
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
        if self.trainer.data_parallel_device_ids is None:
            raise MisconfigurationException('you selected (distribute_backend = ddp) but did not set Trainer(gpus=?)')

        os.environ['PL_TRAINER_GPUS'] = ','.join([str(i) for i in self.trainer.data_parallel_device_ids])
        os.environ['PL_IN_DDP_SUBPROCESS'] = '1'

        if self.trainer.logger is not None:
            os.environ['PL_EXP_VERSION'] = str(self.trainer.logger.version)

        num_gpus = len(self.trainer.data_parallel_device_ids)
        os.environ['WORLD_SIZE'] = f'{num_gpus * self.trainer.num_nodes}'

        self.interactive_ddp_procs = []
        for local_rank in range(1, self.trainer.num_processes):
            env_copy = os.environ.copy()
            env_copy['LOCAL_RANK'] = f'{local_rank}'

            # remove env var if global seed not set
            if os.environ.get('PL_GLOBAL_SEED') is None and 'PL_GLOBAL_SEED' in env_copy:
                del env_copy['PL_GLOBAL_SEED']

            # start process
            # if hydra is available and initialized, make sure to set the cwd correctly
            cwd: Optional[str] = None
            if HYDRA_AVAILABLE:
                if HydraConfig.initialized():
                    cwd = get_original_cwd()
            proc = subprocess.Popen(command, env=env_copy, cwd=cwd)
            self.interactive_ddp_procs.append(proc)

            # starting all processes at once can cause issues
            # with dataloaders delay between 1-10 seconds
            delay = np.random.uniform(1, 5, 1)[0]
            sleep(delay)

    def train(self):
        model = self.trainer.model

        results = self.ddp_train(process_idx=self.task_idx, model=model)
        if 'WORLD_SIZE' in os.environ:
            del os.environ['WORLD_SIZE']
        return results

    def training_step(self, args):
        return self._step(args)

    def validation_step(self, args):
        return self._step(args)

    def test_step(self, args):
        return self._step(args)

    def _step(self, args):
        args = self.ddp_plugin.on_before_forward(self.trainer.get_model(), *args)
        if self.trainer.amp_backend == AMPType.NATIVE:
            with torch.cuda.amp.autocast():
                output = self.trainer.model(*args)
        else:
            output = self.trainer.model(*args)
        return output

    def barrier(self, name: Optional[str] = None):
        if self.rpc_enabled:
            # Allow RPC to handle barrier on main RPC processes
            self.ddp_plugin.barrier()
        elif torch_distrib.is_initialized():
            torch_distrib.barrier(group=self.ddp_plugin.data_parallel_group)

    def _check_can_spawn_children(self):
        if self._has_spawned_children:
            raise RuntimeError(
                "You tried to run `.fit` or `.test` multiple times in the same script."
                " This is not supported in DDP mode, switch to `accelerator='ddp_spawn'` instead."
            )

    def set_world_ranks(self, process_idx):
        self.trainer.local_rank = process_idx
        self.trainer.global_rank = self.trainer.node_rank * self.trainer.num_processes + process_idx
        self.trainer.world_size = self.trainer.num_nodes * self.trainer.num_processes

    def init_device(self, process_idx):
        self.trainer.root_gpu = self.trainer.data_parallel_device_ids[self.trainer.local_rank]
        torch.cuda.set_device(self.trainer.root_gpu)

    def model_to_device(self, model):
        model.cuda(self.trainer.root_gpu)

    def get_device_ids(self):
        device_ids = [self.trainer.root_gpu]
        return device_ids

    def on_train_end(self):
        pass

    def early_stopping_should_stop(self, pl_module):
        stop = torch.tensor(int(self.trainer.should_stop), device=pl_module.device)
        torch_distrib.all_reduce(stop, op=torch_distrib.reduce_op.SUM)
        self.barrier('early_stopping')
        should_stop = stop == self.trainer.world_size
        return should_stop

    def broadcast(self, obj, src=0):
        return self.dist.broadcast(obj, group=self.ddp_plugin.data_parallel_group)

    def ddp_train(self, process_idx, model):
        """
        Entry point for ddp

        Args:
            process_idx:
            mp_queue: multiprocessing queue
            model:

        Returns:
            Dict with evaluation results

        """
        seed = os.environ.get("PL_GLOBAL_SEED")
        if seed is not None:
            seed_everything(int(seed))

        # show progressbar only on progress_rank 0
        if (self.trainer.node_rank != 0 or process_idx != 0) and self.trainer.progress_bar_callback is not None:
            self.trainer.progress_bar_callback.disable()

        # determine which process we are and world size
        self.set_world_ranks(process_idx)

        # set warning rank
        rank_zero_only.rank = self.trainer.global_rank

        # Initialize cuda device
        self.init_device(process_idx)

        # set up server using proc 0's ip address
        # try to init for 20 times at max in case ports are taken
        # where to store ip_table
        model.trainer = self.trainer
        self.init_ddp_connection(
            self.trainer.global_rank,
            self.trainer.world_size,
            self.trainer.is_slurm_managing_tasks
        )

        if isinstance(self.ddp_plugin, RPCPlugin):
            if not self.ddp_plugin.is_main_rpc_process:
                self.ddp_plugin.on_accelerator_exit_rpc_process(self.trainer)
                self.ddp_plugin.exit_rpc_process()
                if self.ddp_plugin.return_after_exit_rpc_process:
                    return
            else:
                self.ddp_plugin.on_main_rpc_connection(self.trainer)

        # call setup after the ddp process has connected
        self.trainer.call_setup_hook(model)

        # on world_size=0 let everyone know training is starting
        if self.trainer.is_global_zero and not torch.distributed.is_initialized():
            log.info('-' * 100)
            log.info(f'distributed_backend={self.trainer.distributed_backend}')
            log.info(f'All DDP processes registered. Starting ddp with {self.trainer.world_size} processes')
            log.info('-' * 100)

        # call sync_bn before .cuda(), configure_apex and configure_ddp
        if self.trainer.sync_batchnorm:
            model = self.configure_sync_batchnorm(model)

        # move the model to the correct device
        self.model_to_device(model)

        # CHOOSE OPTIMIZER
        # allow for lr schedulers as well
        self.setup_optimizers(model)

        # 16-bit
        model = self.trainer.precision_connector.connect(model)

        # device ids change depending on the DDP setup
        device_ids = self.get_device_ids()

        # allow user to configure ddp
        model = self.configure_ddp(model, device_ids)

        self.barrier('ddp_setup')
        self.trainer.setup_trainer(model)

        # train or test
        results = self.train_or_test()

        # clean up memory
        torch.cuda.empty_cache()

        return results

    def configure_ddp(
            self, model: LightningModule, device_ids: List[int]
    ) -> DistributedDataParallel:
        self.ddp_plugin.device_ids = device_ids
        model = self.ddp_plugin.configure_ddp(model, device_ids)
        return model

    def configure_sync_batchnorm(self, model: LightningModule) -> LightningModule:
        """
        Add global batchnorm for a model spread across multiple GPUs and nodes.

        Override to synchronize batchnorm between specific process groups instead
        of the whole world or use a different sync_bn like `apex`'s version.

        Args:
            model: pointer to current :class:`LightningModule`.

        Return:
            LightningModule with batchnorm layers synchronized between process groups
        """
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group=None)

        return model

    def sync_tensor(self,
                    tensor: Union[torch.Tensor],
                    group: Optional[Any] = None,
                    reduce_op: Optional[Union[ReduceOp, str]] = None) -> torch.Tensor:
        """

        """
        return sync_ddp_if_available(tensor, group, reduce_op)

    def all_gather(self, tensor: Union[torch.Tensor], group: Optional[Any] = None, sync_grads: bool = False):
        """
        Function to gather a tensor from several distributed processes

        Args:
            tensor: tensor of shape (batch, ...)
            group: the process group to gather results from. Defaults to all processes (world)
            sync_grads: flag that allows users to synchronize gradients for all_gather op

        Return:
            A tensor of shape (world_size, batch, ...)
        """
        return all_gather_ddp_if_available(tensor, group=group, sync_grads=sync_grads)

    def get_reference_model(self, model) -> LightningModule:
        return self.ddp_plugin.get_model_from_plugin(model)

    @property
    def distributed_sampler_kwargs(self):
        distributed_sampler_kwargs = dict(
            num_replicas=self.trainer.num_nodes * self.trainer.num_processes,
            rank=self.trainer.global_rank
        )
        if self.ddp_plugin is not None:
            distributed_sampler_kwargs = self.ddp_plugin.distributed_sampler_kwargs(distributed_sampler_kwargs)
        return distributed_sampler_kwargs

    @property
    def require_distributed_sampler(self):
        return True
