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
from typing import Optional

import torch
from torch.optim import Optimizer

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.core.optimizer import is_lightning_optimizer
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities import _FAIRSCALE_AVAILABLE, _FAIRSCALE_OSS_FP16_BROADCAST_AVAILABLE, rank_zero_only
from pytorch_lightning.utilities.exceptions import MisconfigurationException

if _FAIRSCALE_AVAILABLE:
    from fairscale.nn.data_parallel.sharded_ddp import ShardedDataParallel
    from fairscale.optim import OSS

    from pytorch_lightning.overrides.fairscale import LightningShardedDataParallel, unwrap_lightning_module_sharded


class DDPShardedPlugin(DDPPlugin):
    """ Optimizer and gradient sharded training provided by FairScale. """

    _REDUCE_BUFFER_SIZE_DEFAULT = 2**23  # 8M

    def configure_ddp(self):
        self._wrap_optimizers()
        self._model = ShardedDataParallel(
            LightningShardedDataParallel(self.model),
            sharded_optimizer=self.lightning_module.trainer.optimizers,
            # For multi-node training, enabling bucketing will improve performance.
            reduce_buffer_size=self._REDUCE_BUFFER_SIZE_DEFAULT if self.num_nodes > 1 else 0,
        )
        setattr(self._model, "require_backward_grad_sync", False)

    def _reinit_optimizers_with_oss(self):
        optimizers = self.lightning_module.trainer.optimizers
        for x, optimizer in enumerate(optimizers):
            if is_lightning_optimizer(optimizer):
                optimizer = optimizer._optimizer
            if not isinstance(optimizer, OSS):
                optim_class = type(optimizer)
                zero_optimizer = OSS(params=optimizer.param_groups, optim=optim_class, **optimizer.defaults)
                if _FAIRSCALE_OSS_FP16_BROADCAST_AVAILABLE:
                    precision = self.lightning_module.trainer.precision
                    is_fp16 = precision in ("mixed", 16)
                    # For multi-node training, compressing the model shards in fp16 before broadcasting
                    # improves performance. When using PyTorch AMP, it will not degrade
                    # the model performance.
                    zero_optimizer.broadcast_fp16 = is_fp16 and self.num_nodes > 1
                optimizers[x] = zero_optimizer
                del optimizer
        trainer = self.lightning_module.trainer
        trainer.optimizers = optimizers
        trainer.convert_to_lightning_optimizers()

    def _wrap_optimizers(self):
        if self.model.trainer.state.fn != TrainerFn.FITTING:
            return
        self._reinit_optimizers_with_oss()

    def optimizer_state(self, optimizer: "OSS") -> Optional[dict]:
        if is_lightning_optimizer(optimizer):
            optimizer = optimizer._optimizer
        optimizer.consolidate_state_dict()
        return self._optim_state_dict(optimizer)

    @rank_zero_only
    def _optim_state_dict(self, optimizer):
        """
        Retrieves state dict only on rank 0, which contains the entire optimizer state after calling
        :meth:`consolidate_state_dict`.
        """
        return optimizer.state_dict()

    @property
    def lightning_module(self) -> LightningModule:
        if not _FAIRSCALE_AVAILABLE:  # pragma: no cover
            raise MisconfigurationException(
                "`DDPShardedPlugin` requires `fairscale` to be installed."
                " Install it by running `pip install fairscale`."
            )
        return unwrap_lightning_module_sharded(self._model)

    def pre_backward(self, closure_loss: torch.Tensor, should_accumulate: bool, optimizer: Optimizer, opt_idx: int):
        pass

    def post_training_step(self):
        pass
