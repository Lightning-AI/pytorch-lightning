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
# limitations under the License.
import logging
import os
import subprocess
import sys
from time import sleep
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.distributed as torch_distrib
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.optim import Optimizer

from pytorch_lightning.distributed import LightningDistributed
from pytorch_lightning.overrides import LightningDistributedModule
from pytorch_lightning.overrides.distributed import prepare_for_backward
from pytorch_lightning.plugins.environments.cluster_environment import ClusterEnvironment
from pytorch_lightning.plugins.training_type.parallel import ParallelPlugin
from pytorch_lightning.utilities import (
    _HYDRA_AVAILABLE,
    _TORCH_GREATER_EQUAL_1_7,
    _TORCH_GREATER_EQUAL_1_8,
    rank_zero_deprecation,
    rank_zero_warn,
)
from pytorch_lightning.utilities.distributed import rank_zero_only, ReduceOp, sync_ddp_if_available
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.seed import reset_seed

if _HYDRA_AVAILABLE:
    from hydra.core.hydra_config import HydraConfig
    from hydra.utils import get_original_cwd, to_absolute_path
if _TORCH_GREATER_EQUAL_1_8:
    from pytorch_lightning.utilities.distributed import register_ddp_comm_hook

log = logging.getLogger(__name__)


class DDPPlugin(ParallelPlugin):
    """
    Plugin for multi-process single-device training on one or multiple nodes.

    The master process in each node spawns N-1 child processes via :func:`subprocess.Popen`,
    where N is the number of devices (e.g. GPU) per node.
    It is very similar to how :mod:`torch.distributed.launch` launches processes.
    """

    distributed_backend = "ddp"

    def __init__(
        self,
        parallel_devices: Optional[List[torch.device]] = None,
        num_nodes: Optional[int] = None,
        cluster_environment: ClusterEnvironment = None,
        sync_batchnorm: Optional[bool] = None,
        ddp_comm_state: Optional[object] = None,
        ddp_comm_hook: Optional[callable] = None,
        ddp_comm_wrapper: Optional[callable] = None,
        **kwargs: Union[Any, Dict[str, Any]],
    ) -> None:
        super().__init__(parallel_devices=parallel_devices, cluster_environment=cluster_environment)
        self.interactive_ddp_procs = []
        if num_nodes is not None:
            rank_zero_deprecation(
                "Argument `num_nodes` in `DDPPlugin` is deprecated in v1.4, and will be removed in v1.6."
                " Notice that it will be overriden by the trainer setting."
            )
        self._num_nodes = num_nodes or 1
        if sync_batchnorm is not None:
            rank_zero_deprecation(
                "Argument `sync_batchnorm` in `DDPPlugin` is deprecated in v1.4, and will be removed in v1.6."
                " Notice that it will be overriden by the trainer setting."
            )
        self._sync_batchnorm = sync_batchnorm or False
        self.dist = LightningDistributed()
        self.num_processes = len(self.parallel_devices) if self.parallel_devices is not None else 0
        self._ddp_kwargs = kwargs
        self._has_spawned_children = False
        self.task_idx = None
        self._ddp_comm_state = ddp_comm_state
        self._ddp_comm_hook = ddp_comm_hook
        self._ddp_comm_wrapper = ddp_comm_wrapper
        self.set_world_ranks()

    @property
    def root_device(self) -> torch.device:
        return self.parallel_devices[self.local_rank]

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @num_nodes.setter
    def num_nodes(self, num_nodes: int) -> None:
        # note that world ranks is related to num_nodes, when resetting it, need to reset world ranks
        self._num_nodes = num_nodes
        self.set_world_ranks()

    @property
    def sync_batchnorm(self) -> bool:
        return self._sync_batchnorm

    @sync_batchnorm.setter
    def sync_batchnorm(self, sync_batchnorm: bool) -> None:
        self._sync_batchnorm = sync_batchnorm

    @property
    def distributed_sampler_kwargs(self):
        distributed_sampler_kwargs = dict(num_replicas=(self.num_nodes * self.num_processes), rank=self.global_rank)
        return distributed_sampler_kwargs

    @property
    def _is_single_process_single_device(self) -> bool:
        return True

    def setup_environment(self) -> None:
        # start the other scripts
        if not self.cluster_environment.creates_children() and os.environ.get("PL_IN_DDP_SUBPROCESS", "0") != "1":
            self._call_children_scripts()

        # set the task idx
        self.task_idx = self.cluster_environment.local_rank()

        self.setup_distributed()

    def _call_children_scripts(self):

        # bookkeeping of spawned processes
        assert self.local_rank == 0
        self._check_can_spawn_children()
        self._has_spawned_children = True

        # DDP Environment variables
        os.environ["MASTER_ADDR"] = self.cluster_environment.master_address()
        os.environ["MASTER_PORT"] = str(self.cluster_environment.master_port())

        # allow the user to pass the node rank
        os.environ["NODE_RANK"] = str(self.cluster_environment.node_rank())
        os.environ["LOCAL_RANK"] = str(self.cluster_environment.local_rank())

        # when user is using hydra find the absolute path
        path_lib = os.path.abspath if not _HYDRA_AVAILABLE else to_absolute_path

        # pull out the commands used to run the script and resolve the abs file path
        command = sys.argv
        try:
            full_path = path_lib(command[0])
        except Exception:
            full_path = os.path.abspath(command[0])

        command[0] = full_path
        # use the same python interpreter and actually running
        command = [sys.executable] + command

        # the visible devices tell us how many GPUs we want to use.
        # when the trainer script was called the device has already been scoped by the time
        # code reaches this point. so, to call the scripts, we need to leave cuda visible devices alone
        # but forward the GPUs selected via environment variables
        if self.parallel_devices is None:
            raise MisconfigurationException("you selected (distribute_backend = ddp) but did not set Trainer(gpus=?)")

        os.environ["PL_TRAINER_GPUS"] = ",".join([str(device.index) for device in self.parallel_devices])
        os.environ["PL_IN_DDP_SUBPROCESS"] = "1"

        num_gpus = len(self.parallel_devices)
        os.environ["WORLD_SIZE"] = f"{num_gpus * self.num_nodes}"

        self.interactive_ddp_procs = []

        for local_rank in range(1, self.num_processes):
            env_copy = os.environ.copy()
            env_copy["LOCAL_RANK"] = f"{local_rank}"

            if self.lightning_module.logger is not None:
                # spawned processes must reference the same log dir, prevent auto-increment version
                env_copy["PL_EXP_VERSION"] = str(self.lightning_module.logger.version)

            # remove env var if global seed not set
            if os.environ.get("PL_GLOBAL_SEED") is None and "PL_GLOBAL_SEED" in env_copy:
                del env_copy["PL_GLOBAL_SEED"]

            # start process
            # if hydra is available and initialized, make sure to set the cwd correctly
            cwd: Optional[str] = None
            if _HYDRA_AVAILABLE:
                if HydraConfig.initialized():
                    cwd = get_original_cwd()
                    os_cwd = f'"{os.getcwd()}"'
                    command += [f'hydra.run.dir={os_cwd}', f'hydra.job.name=train_ddp_process_{local_rank}']
            proc = subprocess.Popen(command, env=env_copy, cwd=cwd)
            self.interactive_ddp_procs.append(proc)

            # starting all processes at once can cause issues
            # with dataloaders delay between 1-10 seconds
            delay = np.random.uniform(1, 5, 1)[0]
            sleep(delay)

    def setup_distributed(self):
        reset_seed()

        # determine which process we are and world size
        self.set_world_ranks()

        # set warning rank
        rank_zero_only.rank = self.global_rank

        # set up server using proc 0's ip address
        # try to init for 20 times at max in case ports are taken
        # where to store ip_table
        self.init_ddp_connection()

        # on world_size=0 let everyone know training is starting
        if self.is_global_zero and not torch.distributed.is_initialized():
            log.info("-" * 100)
            log.info(f"distributed_backend={self.distributed_backend}")
            log.info(f"All DDP processes registered. Starting ddp with {self.world_size} processes")
            log.info("-" * 100)

        # set the ranks and devices
        self.dist.rank = self.global_rank
        self.dist.device = self.root_device

    def _check_can_spawn_children(self):
        if self._has_spawned_children:
            raise RuntimeError(
                "You tried to run `.fit` or `.test` multiple times in the same script."
                " This is not supported in DDP mode, switch to `distributed_backend='ddp_spawn'` instead."
            )

    def set_world_ranks(self) -> None:
        if self.cluster_environment is None:
            return
        self.cluster_environment.set_global_rank(self.node_rank * self.num_processes + self.local_rank)
        self.cluster_environment.set_world_size(self.num_nodes * self.num_processes)
        rank_zero_only.rank = self.cluster_environment.global_rank()

    def pre_configure_ddp(self):
        # if unset, default `find_unused_parameters` `True`
        # Many models require setting this parameter to True, as there are corner cases
        # when not all parameter backward hooks are fired by the autograd engine even if require_grad is set to True.
        # This flag does come with a performance hit, so it is suggested to disable in cases where it is possible.
        self._ddp_kwargs["find_unused_parameters"] = self._ddp_kwargs.get("find_unused_parameters", True)
        # todo: PyTorch 1.7.0 DDP introduces ``self.reducer._rebuild_buckets()`` breaking manual_optimization
        if _TORCH_GREATER_EQUAL_1_7 and not self.lightning_module.automatic_optimization and not self._ddp_kwargs.get(
            "find_unused_parameters", False
        ):
            rank_zero_warn(
                "From PyTorch 1.7.0, Lightning ``manual_optimization`` needs to set ``find_unused_parameters=True`` "
                "to properly work with DDP."
            )
            self._ddp_kwargs["find_unused_parameters"] = True

    def _register_ddp_hooks(self) -> None:
        # currently, DDP communication hooks only work with NCCL backend and SPSD (single process single device) mode
        # https://github.com/pytorch/pytorch/blob/v1.8.0/torch/nn/parallel/distributed.py#L1080-L1084
        if (_TORCH_GREATER_EQUAL_1_8 and self.on_gpu and self._is_single_process_single_device):
            register_ddp_comm_hook(
                model=self._model,
                ddp_comm_state=self._ddp_comm_state,
                ddp_comm_hook=self._ddp_comm_hook,
                ddp_comm_wrapper=self._ddp_comm_wrapper,
            )

    def configure_ddp(self):
        self.pre_configure_ddp()
        self._model = DistributedDataParallel(
            LightningDistributedModule(self.model),
            device_ids=self.determine_ddp_device_ids(),
            **self._ddp_kwargs,
        )
        self._register_ddp_hooks()

    def determine_ddp_device_ids(self):
        if self.root_device.type == "cpu":
            return None
        return [self.root_device.index]

    def init_ddp_connection(self, global_rank: Optional[int] = None, world_size: Optional[int] = None) -> None:
        global_rank = global_rank if global_rank is not None else self.cluster_environment.global_rank()
        world_size = world_size if world_size is not None else self.cluster_environment.world_size()
        os.environ["MASTER_ADDR"] = self.cluster_environment.master_address()
        os.environ["MASTER_PORT"] = str(self.cluster_environment.master_port())
        if not torch.distributed.is_initialized():
            log.info(f"initializing ddp: GLOBAL_RANK: {global_rank}, MEMBER: {global_rank + 1}/{world_size}")
            torch_distrib.init_process_group(self.torch_distributed_backend, rank=global_rank, world_size=world_size)

    def pre_dispatch(self):
        # move the model to the correct device
        self.model_to_device()

        if self.sync_batchnorm:
            self.model = self.configure_sync_batchnorm(self.model)

        self.configure_ddp()

        self.barrier()

    def post_dispatch(self) -> None:
        self.cluster_environment.teardown()

    def barrier(self, *args, **kwargs):
        if torch_distrib.is_available() and torch_distrib.is_initialized():
            torch_distrib.barrier()

    def broadcast(self, obj: object, src: int = 0) -> object:
        return self.dist.broadcast(obj)

    def pre_backward(self, closure_loss: torch.Tensor, should_accumulate: bool, optimizer: Optimizer, opt_idx: int):
        """Run before precision plugin executes backward"""
        if not self.lightning_module.automatic_optimization and self.model.require_backward_grad_sync:
            prepare_for_backward(self.model, closure_loss)

    def model_to_device(self):
        if self.root_device.type == "cuda":
            torch.cuda.set_device(self.root_device)
        self.model.to(self.root_device)

    def reduce(self, tensor, group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = "mean"):
        """
        Reduces a tensor from several distributed processes to one aggregated tensor.

        Args:
            tensor: the tensor to sync and reduce
            group: the process group to gather results from. Defaults to all processes (world)
            reduce_op: the reduction operation. Defaults to 'mean'/'avg'.
                Can also be a string 'sum' to calculate the sum during reduction.

        Return:
            reduced value, except when the input was not a tensor the output remains is unchanged
        """
        if isinstance(tensor, torch.Tensor):
            tensor = sync_ddp_if_available(tensor, group, reduce_op=(reduce_op or "mean"))
        return tensor

    def training_step(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def validation_step(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def test_step(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def predict_step(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def post_training_step(self):
        if not self.lightning_module.automatic_optimization:
            self.model.require_backward_grad_sync = True

    @classmethod
    def register_plugins(cls, plugin_registry: Dict) -> None:
        plugin_registry.register(
            "ddp_find_unused_parameters_false",
            cls,
            description="DDP Plugin with `find_unused_parameters` as False",
            find_unused_parameters=False
        )
