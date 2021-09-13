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
from contextlib import ExitStack
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.plugins.collective.collective_plugin import Collective
from pytorch_lightning.plugins.collective.horovod_collective import HorovodCollective
from pytorch_lightning.plugins.io.checkpoint_plugin import CheckpointIO
from pytorch_lightning.plugins.training_type.parallel import ParallelPlugin
from pytorch_lightning.utilities import _HOROVOD_AVAILABLE
from pytorch_lightning.utilities.distributed import rank_zero_only

if _HOROVOD_AVAILABLE:
    import horovod.torch as hvd


class HorovodPlugin(ParallelPlugin):
    """Plugin for Horovod distributed training integration."""

    def __init__(
        self,
        parallel_devices: Optional[List[torch.device]] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        collective: Optional[Collective] = None,
    ):
        super().__init__(
            parallel_devices=parallel_devices,
            cluster_environment=None,
            checkpoint_io=checkpoint_io,
            collective=collective or HorovodCollective(),
        )
        rank_zero_only.rank = self.global_rank

    @property
    def global_rank(self) -> int:
        return hvd.rank()

    @property
    def local_rank(self) -> int:
        return hvd.local_rank()

    @property
    def world_size(self) -> int:
        return hvd.size()

    @property
    def root_device(self):
        return self.parallel_devices[self.local_rank]

    @property
    def distributed_sampler_kwargs(self):
        distributed_sampler_kwargs = dict(num_replicas=self.world_size, rank=self.global_rank)
        return distributed_sampler_kwargs

    def setup(self) -> None:
        self.model_to_device()
        self.collective.on_gpu = self.on_gpu
        self.collective.local_rank = self.local_rank

    def pre_dispatch(self):

        if not self.lightning_module.trainer.training:
            # no need to setup optimizers
            return

        def _unpack_lightning_optimizer(opt):
            return opt._optimizer if isinstance(opt, LightningOptimizer) else opt

        optimizers = self.lightning_module.trainer.optimizers
        optimizers = [_unpack_lightning_optimizer(opt) for opt in optimizers]

        # Horovod: scale the learning rate by the number of workers to account for
        # increased total batch size
        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= self.world_size

        # Horovod: adjust base LR used by schedulers to match scaled optimizer initial LR
        lr_schedulers = self.lightning_module.trainer.lr_schedulers
        for scheduler in lr_schedulers:
            scheduler = scheduler["scheduler"]
            if isinstance(scheduler, _LRScheduler):
                scheduler.base_lrs = [lr * self.world_size for lr in scheduler.base_lrs]

        # Horovod: broadcast parameters & optimizer state to ensure consistent initialization
        hvd.broadcast_parameters(self.lightning_module.state_dict(), root_rank=0)
        for optimizer in optimizers:
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        self.lightning_module.trainer.accelerator.optimizers = self._wrap_optimizers(optimizers)

    def start_training(self, trainer):
        with ExitStack() as stack:
            for optimizer in trainer.optimizers:
                # Synchronization will be performed explicitly following backward()
                stack.enter_context(optimizer.skip_synchronize())

            # set up training routine
            self._results = trainer.run_stage()

        # Make sure all workers have finished training before returning to the user
        self.collective.join()

    def start_evaluating(self, trainer):
        with ExitStack():
            self._results = trainer.run_stage()

        # Make sure all workers have finished training before returning to the user
        self.collective.join()

    def start_predicting(self, trainer):
        with ExitStack():
            # set up training routine
            self._results = trainer.run_stage()

        # Make sure all workers have finished training before returning to the user
        self.collective.join()

    def model_to_device(self):
        if self.on_gpu:
            # this can potentially be removed after #8312. Not done due to lack of horovod testing
            torch.cuda.set_device(self.root_device)
        self.model.to(self.root_device)

    def post_backward(self, closure_loss: torch.Tensor) -> None:
        # synchronize all horovod optimizers.
        for optimizer in self.lightning_module.trainer.optimizers:
            optimizer.synchronize()

    def _wrap_optimizers(self, optimizers: List[Optimizer]) -> List["hvd.DistributedOptimizer"]:
        """Wraps optimizers to perform gradient aggregation via allreduce."""
        return [
            hvd.DistributedOptimizer(opt, named_parameters=self._filter_named_parameters(self.lightning_module, opt))
            if "horovod" not in str(opt.__class__)
            else opt
            for opt in optimizers
        ]

    @staticmethod
    def _filter_named_parameters(model: nn.Module, optimizer: Optimizer) -> List[Tuple[str, nn.Parameter]]:
        opt_params = {p for group in optimizer.param_groups for p in group.get("params", [])}
        return [(name, p) for name, p in model.named_parameters() if p in opt_params]

    def teardown(self) -> None:
        if self.on_gpu:
            # GPU teardown
            self.lightning_module.cpu()
            # clean up memory
            torch.cuda.empty_cache()
