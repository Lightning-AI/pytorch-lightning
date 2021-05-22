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
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, List, Optional

import torch
from torch.nn.parallel import DistributedDataParallel

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.overrides.base import unwrap_lightning_module
from pytorch_lightning.plugins.environments.cluster_environment import ClusterEnvironment
from pytorch_lightning.plugins.training_type.training_type_plugin import TrainingTypePlugin
from pytorch_lightning.utilities import _XLA_AVAILABLE
from pytorch_lightning.utilities.distributed import all_gather_ddp_if_available, ReduceOp


class ParallelPlugin(TrainingTypePlugin, ABC):
    """ Plugin for training with multiple processes in parallel. """

    def __init__(
        self,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
    ):
        super().__init__()
        self.parallel_devices = parallel_devices
        self.cluster_environment = cluster_environment

    @property
    @abstractmethod
    def root_device(self) -> torch.device:
        raise NotImplementedError

    @property
    def on_gpu(self) -> bool:
        return self.root_device.type == "cuda" and torch.cuda.is_available()

    @property
    def on_tpu(self) -> bool:
        return self.root_device.type == "xla" and _XLA_AVAILABLE

    @property
    def lightning_module(self):
        return unwrap_lightning_module(self._model)

    @property
    def global_rank(self) -> int:
        return self.cluster_environment.global_rank() if self.cluster_environment is not None else 0

    @property
    def local_rank(self) -> int:
        return self.cluster_environment.local_rank() if self.cluster_environment is not None else 0

    @property
    def node_rank(self) -> int:
        return self.cluster_environment.node_rank() if self.cluster_environment is not None else 0

    @property
    def world_size(self) -> int:
        return self.cluster_environment.world_size() if self.cluster_environment is not None else 1

    @property
    def is_global_zero(self) -> bool:
        return self.global_rank == 0

    @property
    def distributed_sampler_kwargs(self):
        distributed_sampler_kwargs = dict(num_replicas=len(self.parallel_devices), rank=self.global_rank)
        return distributed_sampler_kwargs

    def all_gather(self, tensor: torch.Tensor, group: Optional[Any] = None, sync_grads: bool = False) -> torch.Tensor:
        """Perform a all_gather on all processes """
        return all_gather_ddp_if_available(tensor, group=group, sync_grads=sync_grads)

    def reduce_boolean_decision(self, decision: bool) -> bool:
        decision = torch.tensor(int(decision), device=self.lightning_module.device)
        decision = self.reduce(decision, reduce_op=ReduceOp.SUM)
        decision = bool(decision == self.world_size)
        return decision

    @property
    def torch_distributed_backend(self):
        torch_backend = os.getenv("PL_TORCH_DISTRIBUTED_BACKEND")
        if torch_backend is None:
            torch_backend = "nccl" if self.on_gpu else "gloo"
        return torch_backend

    @staticmethod
    def configure_sync_batchnorm(model: LightningModule) -> LightningModule:
        """
        Add global batchnorm for a model spread across multiple GPUs and nodes.

        Override to synchronize batchnorm between specific process groups instead
        of the whole world or use a different sync_bn like `apex`'s version.

        Args:
            model: pointer to current :class:`LightningModule`.

        Return:
            LightningModule with batchnorm layers synchronized between process groups
        """
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        return model

    @contextmanager
    def block_backward_sync(self):
        """
        Blocks ddp sync gradients behaviour on backwards pass.
        This is useful for skipping sync when accumulating gradients, reducing communication overhead
        Returns: context manager with sync behaviour off
        """
        if isinstance(self.model, DistributedDataParallel):
            with self.model.no_sync():
                yield None
        else:
            yield None

    def teardown(self) -> None:
        if self.on_gpu:
            # GPU teardown
            self.lightning_module.cpu()
            # clean up memory
            with torch.cuda.device(self.root_device):
                torch.cuda.empty_cache()
