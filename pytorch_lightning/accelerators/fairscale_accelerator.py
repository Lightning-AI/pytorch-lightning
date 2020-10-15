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
from typing import Dict, Any

from fairscale.optim import OSS

from pytorch_lightning.accelerators import DDPAccelerator
from pytorch_lightning.utilities import rank_zero_only


class LightningOSS(OSS):

    @rank_zero_only
    def state_dict(self) -> Dict[str, Any]:
        """
        Ensure we only call state_dict using rank zero.
        """

        assert (
                len(self._all_states) > 0
        ), "The optimizer state is not materialized, please call consolidate_state_dict on every replica beforehand"

        return {"state": self._all_states}


class FairScaleAccelerator(DDPAccelerator):

    def __init__(self, trainer, cluster_environment=None):
        super().__init__(trainer, cluster_environment)
        self.nickname = 'fairscale'

    def setup_optimizers(self, model):
        if self.trainer.testing is True:
            return

        optimizers, lr_schedulers, optimizer_frequencies = self.trainer.init_optimizers(model)
        self.trainer.optimizers = self.re_init_with_fairscale_zero(optimizers)
        self.trainer.lr_schedulers = lr_schedulers
        self.trainer.optimizer_frequencies = optimizer_frequencies

    def re_init_with_fairscale_zero(self, optimizers):
        """
        Re-initialise optimizers to use OSS wrapper. We need to re-initialise due to
        the parameters being sharded across distributed processes, each optimizing a partition.
        Args:
            optimizers: Input optimizers for trainer.
        Returns: Optimizers re-initialised using FairScale OSS (ZERO optimizer).

        """
        fairscale_zero_optimizers = []
        for optimizer in optimizers:
            optim_class = type(optimizer)
            zero_optimizer = LightningOSS(
                params=optimizer.param_groups,
                optim=optim_class,
                **optimizer.defaults
            )
            fairscale_zero_optimizers.append(zero_optimizer)
            del optimizer
        return fairscale_zero_optimizers

    def sync_optim_state(self):
        for optimizer in self.trainer.optimizers:
            optimizer.consolidate_state_dict()
