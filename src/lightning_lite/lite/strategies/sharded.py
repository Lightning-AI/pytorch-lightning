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
from typing import Dict, Generator, List, Optional, Tuple, Union

from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from lightning_lite.lite.strategies.ddp import DDPStrategy
from lightning_lite.lite.utilities.enums import PrecisionType
from lightning_lite.lite.utilities.imports import _FAIRSCALE_AVAILABLE, _FAIRSCALE_OSS_FP16_BROADCAST_AVAILABLE

if _FAIRSCALE_AVAILABLE:
    from fairscale.nn.data_parallel.sharded_ddp import ShardedDataParallel
    from fairscale.optim import OSS
else:
    OSS = ShardedDataParallel = object


class DDPShardedStrategy(DDPStrategy):
    """Optimizer and gradient sharded training provided by FairScale."""

    strategy_name = "ddp_sharded"
    _REDUCE_BUFFER_SIZE_DEFAULT: int = 2**23  # 8M

    def setup_module_and_optimizers(
        self, module: Module, optimizers: List[Optimizer]
    ) -> Tuple[Module, List[Optimizer]]:
        """Wraps the model and optimizers with fairscale components.

        Return:
            The model wrapped into a :class:`~fairscale.nn.data_parallel.ShardedDataParallel` module
            and a list of optimizer wrapped in :class:~`fairscale.optim.OSS`.
        """
        optimizers = self._reinit_optimizers_with_oss(optimizers)
        model = ShardedDataParallel(module, sharded_optimizer=optimizers, **self._ddp_kwargs)
        return model, optimizers

    def pre_backward(self, tensor: Tensor, module: Optional[Module]) -> None:
        pass

    @contextmanager
    def block_backward_sync(self, module: Module) -> Generator:
        """Blocks syncing gradients behaviour on backwards pass.

        This is useful for skipping sync when accumulating gradients, reducing communication overhead
        Returns: context manager with sync behaviour off
        """
        if isinstance(module, ShardedDataParallel):
            with module.no_sync():
                yield None
        else:
            yield None

    @classmethod
    def register_strategies(cls, strategy_registry: Dict) -> None:
        strategy_registry.register(
            "ddp_sharded_find_unused_parameters_false",
            cls,
            description="DDP Sharded Strategy with `find_unused_parameters` as False",
            find_unused_parameters=False,
        )
        strategy_registry.register(
            cls.strategy_name,
            cls,
            description=f"{cls.__class__.__name__}",
        )

    def _reinit_optimizers_with_oss(self, optimizers: List[Union[Optimizer]]) -> List["OSS"]:
        for x, optimizer in enumerate(optimizers):
            if not isinstance(optimizer, OSS):
                optim_class = type(optimizer)
                zero_optimizer = OSS(params=optimizer.param_groups, optim=optim_class, **optimizer.defaults)
                if _FAIRSCALE_OSS_FP16_BROADCAST_AVAILABLE:
                    is_fp16 = self.precision_plugin.precision in (PrecisionType.MIXED, PrecisionType.HALF)
                    # For multi-node training, compressing the model shards in fp16 before broadcasting
                    # improves performance. When using PyTorch AMP, it will not degrade
                    # the model performance.
                    zero_optimizer.broadcast_fp16 = is_fp16 and self.num_nodes > 1
                optimizers[x] = zero_optimizer
                del optimizer
        return optimizers
