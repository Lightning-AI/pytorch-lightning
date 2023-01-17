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
from typing import Any, Dict, Generator, List, Tuple

from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

import pytorch_lightning as pl
from lightning_fabric.utilities.optimizer import _optimizers_to_device
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.overrides.base import _LightningModuleWrapperBase, _LightningPrecisionModuleWrapperBase
from pytorch_lightning.overrides.fairscale import _FAIRSCALE_AVAILABLE, _reinit_optimizers_with_oss
from pytorch_lightning.strategies.ddp_spawn import DDPSpawnStrategy
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation

if _FAIRSCALE_AVAILABLE:
    from fairscale.nn.data_parallel.sharded_ddp import ShardedDataParallel
    from fairscale.optim import OSS

else:
    OSS = ShardedDataParallel = object


class DDPSpawnShardedStrategy(DDPSpawnStrategy):
    """Optimizer sharded training provided by FairScale."""

    strategy_name = "ddp_sharded_spawn"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        rank_zero_deprecation(
            "PyTorch Lightning's sharded implementation using FairScale has been deprecated in v1.9.0 and will be"
            " removed in v2.0.0. You can try using the `Trainer(strategy='fsdp_native')` instead."
            " The difference is that native FSDP uses PyTorch's implementation and the current strategy uses"
            " FairScale's implementation (which was upstreamed to PyTorch). After removal, `strategy='fsdp'` will use"
            " the native version by default."
        )
        super().__init__(*args, **kwargs)

    def connect(self, model: "pl.LightningModule") -> None:
        if not _FAIRSCALE_AVAILABLE:  # pragma: no cover
            raise MisconfigurationException(
                "`DDPSpawnShardedStrategy` requires `fairscale` to be installed."
                " Install it by running `pip install fairscale`."
            )
        return super().connect(model)

    def configure_ddp(self) -> None:
        # set up optimizers after the wrapped module has been moved to the device
        assert self.lightning_module is not None
        self.setup_optimizers(self.lightning_module.trainer)
        assert isinstance(self.model, (pl.LightningModule, _LightningPrecisionModuleWrapperBase))
        self.model, self.optimizers = self._setup_model_and_optimizers(
            model=_LightningModuleWrapperBase(self.model), optimizers=self.optimizers
        )
        _optimizers_to_device(self.optimizers, self.root_device)

    def _setup_model_and_optimizers(self, model: Module, optimizers: List[Optimizer]) -> Tuple[Module, List[Optimizer]]:
        """Wraps the model and optimizers with fairscale components.

        Return:
            The model wrapped into a :class:`~fairscale.nn.data_parallel.ShardedDataParallel` module
            and a list of optimizer wrapped in :class:~`fairscale.optim.OSS`.
        """
        optimizers = self._wrap_optimizers(optimizers)
        model = ShardedDataParallel(model, sharded_optimizer=optimizers, **self._ddp_kwargs)
        return model, optimizers

    def _wrap_optimizers(self, optimizers: List[Optimizer]) -> List["OSS"]:
        assert self.lightning_module
        if self.model is not None and self.lightning_module.trainer.state.fn != TrainerFn.FITTING:
            return optimizers
        optimizers = [o._optimizer if isinstance(o, LightningOptimizer) else o for o in optimizers]
        return _reinit_optimizers_with_oss(optimizers, self.precision_plugin, self.num_nodes)

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

    def pre_backward(self, closure_loss: Tensor) -> None:
        pass

    def post_training_step(self) -> None:
        pass

    @classmethod
    def register_strategies(cls, strategy_registry: Dict) -> None:
        strategy_registry.register(
            "ddp_sharded_spawn_find_unused_parameters_false",
            cls,
            description="DDP Spawn Sharded Strategy with `find_unused_parameters` as False",
            find_unused_parameters=False,
        )
        strategy_registry.register(
            cls.strategy_name,
            cls,
            description=f"{cls.__class__.__name__}",
        )
