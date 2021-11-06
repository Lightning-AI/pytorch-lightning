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
from contextlib import contextmanager
from multiprocessing.queues import SimpleQueue
from typing import Dict, Generator, List, Optional, Tuple

import torch
from torch.nn import Module
from torch.optim import Optimizer

import pytorch_lightning as pl
from pytorch_lightning.plugins.precision.sharded_native_amp import ShardedNativeMixedPrecisionPlugin
from pytorch_lightning.plugins.training_type.ddp_spawn import DDPSpawnPlugin
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities import _FAIRSCALE_AVAILABLE, rank_zero_only
from pytorch_lightning.utilities.enums import DistributedType
from pytorch_lightning.utilities.exceptions import MisconfigurationException

if _FAIRSCALE_AVAILABLE:
    from fairscale.nn.data_parallel.sharded_ddp import ShardedDataParallel
    from fairscale.optim import OSS
    from fairscale.optim.grad_scaler import ShardedGradScaler

    from pytorch_lightning.overrides.fairscale import LightningShardedDataParallel, unwrap_lightning_module_sharded


class DDPSpawnShardedPlugin(DDPSpawnPlugin):
    """Optimizer sharded training provided by FairScale."""

    distributed_backend = DistributedType.DDP_SHARDED_SPAWN

    def configure_ddp(self) -> None:
        trainer = self.lightning_module.trainer
        self._model, optimizers = self._setup_model_and_optimizers(
            model=LightningShardedDataParallel(self.model),
            optimizers=trainer.optimizers,
        )
        trainer.optimizers = optimizers

    def _setup_model_and_optimizers(self, model: Module, optimizers: List[Optimizer]) -> Tuple[Module, List[Optimizer]]:
        """Wraps the model and optimizers with fairscale components.

        Return:
            The model wrapped into a :class:`~fairscale.nn.data_parallel.ShardedDataParallel` module
            and a list of optimizer wrapped in :class:~`fairscale.optim.OSS`.
        """
        optimizers = self._wrap_optimizers(optimizers)
        model = ShardedDataParallel(model, sharded_optimizer=optimizers, **self._ddp_kwargs)
        return model, optimizers

    def _reinit_optimizers_with_oss(self, optimizers: List[Optimizer]) -> List["OSS"]:
        for x, optimizer in enumerate(optimizers):
            if not isinstance(optimizer, OSS):
                optim_class = type(optimizer)
                zero_optimizer = OSS(params=optimizer.param_groups, optim=optim_class, **optimizer.defaults)
                optimizers[x] = zero_optimizer
                del optimizer
        return optimizers

    def _wrap_optimizers(self, optimizers: List[Optimizer]) -> List["OSS"]:
        if self.model is not None and self.model.trainer.state.fn != TrainerFn.FITTING:
            return optimizers

        return self._reinit_optimizers_with_oss(optimizers)

    def optimizer_state(self, optimizer: "OSS") -> Optional[dict]:
        if isinstance(optimizer, OSS):
            optimizer.consolidate_state_dict()
        return self._optim_state_dict(optimizer)

    @contextmanager
    def block_backward_sync(self) -> Generator:
        """Blocks syncing gradients behaviour on backwards pass.

        This is useful for skipping sync when accumulating gradients, reducing communication overhead
        Returns: context manager with sync behaviour off
        """
        if isinstance(self.model, ShardedDataParallel):
            with self.model.no_sync():
                yield None
        else:
            yield None

    @rank_zero_only
    def _optim_state_dict(self, optimizer):
        """
        Retrieves state dict only on rank 0, which contains the entire optimizer state after calling
        :meth:`consolidate_state_dict`.
        """
        return optimizer.state_dict()

    @property
    def lightning_module(self) -> "pl.LightningModule":
        if not _FAIRSCALE_AVAILABLE:  # pragma: no cover
            raise MisconfigurationException(
                "`DDPSpawnShardedPlugin` requires `fairscale` to be installed."
                " Install it by running `pip install fairscale`."
            )
        return unwrap_lightning_module_sharded(self._model)

    def pre_backward(self, closure_loss: torch.Tensor) -> None:
        pass

    def post_training_step(self):
        pass

    def new_process(self, trainer: "pl.Trainer", mp_queue: SimpleQueue) -> None:
        # Ensure that the scaler points to the correct process group
        # which is re-initialized in a new process
        precision_plugin = trainer.accelerator.precision_plugin
        if isinstance(precision_plugin, ShardedNativeMixedPrecisionPlugin):
            precision_plugin.scaler = ShardedGradScaler()
        return super().new_process(trainer, mp_queue)

    @classmethod
    def register_plugins(cls, plugin_registry: Dict) -> None:
        plugin_registry.register(
            "ddp_sharded_spawn_find_unused_parameters_false",
            cls,
            description="DDP Spawn Sharded Plugin with `find_unused_parameters` as False",
            find_unused_parameters=False,
        )
