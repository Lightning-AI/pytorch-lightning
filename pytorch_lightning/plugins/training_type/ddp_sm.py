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
import os
from typing import Any, Dict, Optional, Union

import torch

from pytorch_lightning import _logger as log
from pytorch_lightning.distributed import SMLightningDistributed
from pytorch_lightning.overrides import LightningDistributedModule
from pytorch_lightning.overrides.base import _LightningModuleWrapperBase
from pytorch_lightning.plugins.environments.smdist_environment import SMDistributedEnvironment
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from pytorch_lightning.utilities import _SMDIST_AVAILABLE
from pytorch_lightning.utilities.distributed import rank_zero_only, ReduceOp
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.seed import seed_everything

if _SMDIST_AVAILABLE:
    import smdistributed.dataparallel.torch.distributed as dist
    from smdistributed.dataparallel.torch.parallel.distributed import DistributedDataParallel


class DDPSMPlugin(DDPPlugin):

    distributed_backend = "ddp_sm"

    def __init__(
        self,
        cluster_environment: Optional[SMDistributedEnvironment] = None,
        sync_batchnorm: bool = False,
        **kwargs: Union[Any, Dict[str, Any]],
    ) -> None:
        if not _SMDIST_AVAILABLE:
            raise MisconfigurationException(
                "`smdistributed` module is not available."
                " You would need to enable distributed=smdistributed"
                " in the Sagemaker Estimator Object."
            )

        # While running smdistributed, all the gpus in the instance are considered
        parallel_device_ids = list(range(torch.cuda.device_count()))
        self.parallel_devices = [torch.device("cuda", i) for i in parallel_device_ids]
        num_nodes = len(os.environ['SM_HOSTS'].split(","))

        super().__init__(
            parallel_devices=self.parallel_devices,
            num_nodes=num_nodes,
            cluster_environment=cluster_environment,
            sync_batchnorm=sync_batchnorm
        )

        self._ddp_kwargs = kwargs
        self.dist = SMLightningDistributed()

    def setup(self, model):
        self._model = model

        self.node_rank = self.cluster_environment.node_rank()
        self.local_rank = self.cluster_environment.local_rank()
        self.global_rank = self.cluster_environment.global_rank()
        self.world_size = self.cluster_environment.world_size()

        rank_zero_only.rank = self.global_rank
        self.model_to_device()

    def configure_ddp(self):
        self.pre_configure_ddp()
        self._model = DistributedDataParallel(
            LightningDistributedModule(self.model),
            device_ids=[dist.get_local_rank()],
            broadcast_buffers=False,
        )

    def init_ddp_connection(self, global_rank: int, world_size: int) -> None:

        if not dist.is_initialized():
            log.info(f"initializing ddp: GLOBAL_RANK: {global_rank}, MEMBER: {global_rank + 1}/{world_size}")
            dist.init_process_group(self.torch_distributed_backend, rank=global_rank, world_size=world_size)

    def pre_dispatch(self):
        # TODO: check if needed
        seed = os.environ.get("PL_GLOBAL_SEED")
        if seed is not None:
            seed_everything(int(seed))

        # set warning rank
        rank_zero_only.rank = self.global_rank

        # set up server using proc 0's ip address
        # try to init for 20 times at max in case ports are taken
        # where to store ip_table
        self.init_ddp_connection(self.global_rank, self.world_size)

        # TODO: we moved it to the trainer.fit after calling pre_dispatch
        #   ... need to double check that it is the correct place
        # self.trainer.call_setup_hook(self.model)

        # on world_size=0 let everyone know training is starting
        if self.is_global_zero and not dist.is_initialized():
            log.info("-" * 100)
            log.info(f"distributed_backend={self.distributed_backend}")
            log.info(f"All DDP processes registered. Starting ddp with {self.world_size} processes")
            log.info("-" * 100)

        # # set the ranks and devices
        self.dist.rank = self.global_rank
        self.dist.device = self.root_device

        if self.sync_batchnorm:
            self.model = self.configure_sync_batchnorm(self.model)

        # move the model to the correct device
        self.model_to_device()

        self.configure_ddp()

        self.barrier("configure ddp")

    def sync_ddp_if_available(
        self,
        result: Union[torch.Tensor],
        group: Optional[Any] = None,
        reduce_op: Optional[Union[ReduceOp, str]] = None
    ) -> torch.Tensor:
        """
        Function to reduce a tensor across worker processes during distributed training
        Args:
            result: the value to sync and reduce (typically tensor or number)
            group: the process group to gather results from. Defaults to all processes (world)
            reduce_op: the reduction operation. Defaults to sum.
                Can also be a string of 'avg', 'mean' to calculate the mean during reduction.

        Return:
            reduced value
        """
        if dist.is_available() and dist.is_initialized():
            return self.sync_ddp(result, group=group, reduce_op=reduce_op)
        return result

    def sync_ddp(
        self,
        result: Union[torch.Tensor],
        group: Optional[Any] = None,
        reduce_op: Optional[Union[ReduceOp, str]] = None
    ) -> torch.Tensor:
        """
        Function to reduce the tensors from several ddp processes to one master process

        Args:
            result: the value to sync and reduce (typically tensor or number)
            group: the process group to gather results from. Defaults to all processes (world)
            reduce_op: the reduction operation. Defaults to sum.
                Can also be a string of 'avg', 'mean' to calculate the mean during reduction.

        Return:
            reduced value
        """
        divide_by_world_size = False

        if group is None:
            group = dist.group.WORLD

        op = reduce_op if isinstance(reduce_op, ReduceOp) else ReduceOp.SUM

        if isinstance(reduce_op, str) and reduce_op.lower() in ("avg", "mean"):
            divide_by_world_size = True

        # sync all processes before reduction
        dist.barrier(group=group)
        dist.all_reduce(result, op=op, group=group, async_op=False)

        if divide_by_world_size:
            result = result / dist.get_world_size(group)

        return result

    @property
    def lightning_module(self):
        return self.unwrap_lightning_module()

    def unwrap_lightning_module(self):
        model = self._model
        if isinstance(model, (DistributedDataParallel)):
            model = model.module
        if isinstance(model, _LightningModuleWrapperBase):
            model = model.module
        return model

    def barrier(self, *args, **kwargs) -> None:
        if dist.is_initialized():
            dist.barrier()

    def broadcast(self, obj: object, src: int = 0) -> object:
        return self.dist.broadcast(obj)

    def reduce(self, tensor, group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = "mean"):
        """
        Reduces a tensor from several distributed processes to one aggregated tensor.
        As this plugin only operates with a single device, the reduction is simply the identity.

        Args:
            tensor: the tensor to sync and reduce
            *args: ignored
            **kwargs: ignored

        Return:
            the unmodified input as reduction is not needed for single process operation
        """
        if isinstance(tensor, torch.Tensor):
            tensor = self.sync_ddp_if_available(tensor, group, reduce_op=(reduce_op or "mean"))
        return tensor
