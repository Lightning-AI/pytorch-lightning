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
from typing import Any, List, Optional, Union

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.core.optimizer import is_lightning_optimizer
from pytorch_lightning.plugins.ddp_plugin import DDPPlugin
from pytorch_lightning.plugins.sharded_native_amp_plugin import ShardedNativeAMPPlugin
from pytorch_lightning.utilities import AMPType, FAIRSCALE_AVAILABLE, rank_zero_only
from pytorch_lightning.utilities.exceptions import MisconfigurationException

if FAIRSCALE_AVAILABLE:
    from fairscale.optim import OSS

    from pytorch_lightning.overrides.fairscale import LightningShardedDataParallel


class DDPShardedPlugin(DDPPlugin):

    def __init__(self, **kwargs):
        self._check_fairscale()
        super().__init__(**kwargs)

    def configure_ddp(
            self, model: LightningModule, device_ids: List[int]
    ):
        self._wrap_optimizers(model)
        return LightningShardedDataParallel(model, sharded_optimizer=model.trainer.optimizers)

    def optimizer_state(self, optimizer: 'OSS') -> Optional[dict]:
        optimizer.consolidate_state_dict()
        return self._optim_state_dict(optimizer)

    def _check_fairscale(self):
        if not FAIRSCALE_AVAILABLE:
            raise MisconfigurationException(
                'Sharded DDP Plugin requires Fairscale to be installed.'
            )

    @rank_zero_only
    def _optim_state_dict(self, optimizer):
        return optimizer.state_dict()

    def _wrap_optimizers(self, model):
        trainer = model.trainer
        if trainer.testing is True:
            return

        self._reinit_with_fairscale_oss(trainer)

    def _reinit_with_fairscale_oss(self, trainer):
        optimizers = trainer.optimizers
        for x, optimizer in enumerate(optimizers):
            if is_lightning_optimizer(optimizer):
                optimizer = optimizer.optimizer
            if not isinstance(optimizer, OSS):
                optim_class = type(optimizer)
                zero_optimizer = OSS(
                    params=optimizer.param_groups,
                    optim=optim_class,
                    **optimizer.defaults
                )
                optimizers[x] = zero_optimizer
                del optimizer

    def get_model_from_plugin(
            self,
            model: Union['LightningShardedDataParallel', LightningModule]
    ) -> LightningModule:
        if isinstance(model, LightningShardedDataParallel):
            return model.module
        return model

    def required_plugins(self, amp_backend: AMPType, trainer) -> list:
        if amp_backend == AMPType.APEX:
            raise MisconfigurationException(
                'Sharded Plugin is not supported with Apex AMP, please using native AMP for 16-bit precision.'
            )
        if amp_backend == AMPType.NATIVE:
            return [ShardedNativeAMPPlugin(trainer=trainer)]
        return []

    def on_before_manual_backward(self, model: 'LightningShardedDataParallel', output: Any):
        pass

    def on_after_manual_backward(self, model: 'LightningShardedDataParallel'):
        pass
