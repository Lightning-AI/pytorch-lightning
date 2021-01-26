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
from typing import Any, Callable, Optional, Union

import torch

from pytorch_lightning.accelerators.accelerator import Accelerator, ReduceOp
from pytorch_lightning.cluster_environments import ClusterEnvironment
from pytorch_lightning.utilities import AMPType, rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class CPUAccelerator(Accelerator):

    def __init__(self, trainer, cluster_environment: Optional[ClusterEnvironment] = None):
        """
        Runs training on CPU

        Example::

            # default
            trainer = Trainer(accelerator=CPUAccelerator())

        """
        super().__init__(trainer, cluster_environment)
        self.nickname = None

    def setup(self, model):
        # run through amp wrapper
        if self.trainer.amp_backend:
            raise MisconfigurationException('amp + cpu is not supported.  Please use a GPU option')

        # call setup after the ddp process has connected
        self.trainer.call_setup_hook(model)

        # CHOOSE OPTIMIZER
        # allow for lr schedulers as well
        self.setup_optimizers(model)

        self.trainer.model = model

    def _step(self, model_step: Callable, args):
        if self.trainer.amp_backend == AMPType.NATIVE:
            with torch.cuda.amp.autocast():
                output = model_step(*args)
        else:
            output = model_step(*args)
        return output

    def training_step(self, args):
        return self._step(self.trainer.model.training_step, args)

    def validation_step(self, args):
        return self._step(self.trainer.model.validation_step, args)

    def test_step(self, args):
        return self._step(self.trainer.model.test_step, args)

    def sync_tensor(self,
                    tensor: Union[torch.Tensor],
                    group: Optional[Any] = None,
                    reduce_op: Optional[Union[ReduceOp, str]] = None) -> torch.Tensor:
        return tensor

    @property
    def require_distributed_sampler(self):
        return False
