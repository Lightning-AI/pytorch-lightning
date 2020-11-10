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
from typing import List, cast

from pytorch_lightning import LightningModule
from pytorch_lightning.overrides.fairscale import LightningOSS, LightningShardedDataParallel
from pytorch_lightning.overrides.pytorch import ShardedGradScaler
from pytorch_lightning.plugins.ddp_plugin import DDPPlugin
from pytorch_lightning.utilities import AMPType


class ShardedDDPPlugin(DDPPlugin):

    def configure_ddp(
            self, model: LightningModule, device_ids: List[int]
    ):
        self._setup_optimizers_and_scaler(model)
        if model.trainer.testing:  # Revert to standard DDP if using testing
            super().configure_ddp(
                model=model,
                device_ids=device_ids
            )
        else:
            model = LightningShardedDataParallel(model, sharded_optimizer=model.trainer.optimizers)
        return model

    def sync_optim_state(self, model: LightningModule):
        for optimizer in model.trainer.optimizers:
            optimizer.consolidate_state_dict()

    def sync_backward(self, model: LightningModule):
        # Ensure all backward handles have been called before calling optimizer step
        model = cast(LightningShardedDataParallel, model)
        model.clear_backward_handles()

    def _setup_optimizers_and_scaler(self, model):
        trainer = model.trainer
        if trainer.testing is True:
            return

        optimizers, lr_schedulers, optimizer_frequencies = trainer.init_optimizers(model)
        trainer.optimizers = self._re_init_with_fairscale_zero(optimizers, lr_schedulers)
        trainer.lr_schedulers = lr_schedulers
        trainer.optimizer_frequencies = optimizer_frequencies
        if trainer.amp_backend == AMPType.NATIVE:
            trainer.scaler = ShardedGradScaler()

    def _re_init_with_fairscale_zero(self, optimizers, lr_schedulers):
        """
        Re-initialise optimizers to use OSS wrapper. We need to re-initialise due to
        the parameters being sharded across distributed processes, each optimizing a partition.
        Args:
            optimizers: Input optimizers for trainer.
        Returns: Optimizers re-initialised using FairScale OSS (ZERO optimizer).

        """
        fairscale_zero_optimizers = []
        for optimizer in optimizers:
            if not isinstance(optimizer, LightningOSS):
                optim_class = type(optimizer)
                zero_optimizer = LightningOSS(
                    params=optimizer.param_groups,
                    optim=optim_class,
                    **optimizer.defaults
                )
                fairscale_zero_optimizers.append(zero_optimizer)
                for scheduler in lr_schedulers:
                    scheduler = scheduler['scheduler']
                    if scheduler.optimizer == optimizer:
                        scheduler.optimizer = zero_optimizer
                del optimizer
        return fairscale_zero_optimizers
