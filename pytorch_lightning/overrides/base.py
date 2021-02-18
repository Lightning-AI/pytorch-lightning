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
from typing import Any

import torch
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.warnings import WarningCache

warning_cache = WarningCache()


class _LightningModuleWrapperBase(torch.nn.Module):

    def __init__(self, pl_module: LightningModule):
        """
        Wraps the user's LightningModule and redirects the forward call to the appropriate
        method, either ``training_step``, ``validation_step`` or ``test_step``.
        If the LightningModule is in none of the states `training`, `testing` or `validation`,
        the inputs will be redirected to the
        :meth:`~pytorch_lightning.core.lightning.LightningModule.predict` method.
        Inheriting classes may also modify the inputs or outputs of forward.

        Args:
            pl_module: the model to wrap
        """
        super().__init__()
        self.module = pl_module

    def forward(self, *inputs, **kwargs):
        running_stage = self.module.running_stage

        if running_stage == RunningStage.TRAINING:
            output = self.module.training_step(*inputs, **kwargs)

            # In manual_optimization, we need to prevent DDP reducer as
            # it is done manually in ``LightningModule.manual_backward``
            # `require_backward_grad_sync` will be reset in the
            # ddp_plugin ``post_training_step`` hook
            if not self.module.automatic_optimization:
                self.module.trainer.model.require_backward_grad_sync = False
            warn_if_output_is_none(output, "training_step")

        elif running_stage == RunningStage.TESTING:
            output = self.module.test_step(*inputs, **kwargs)
            warn_if_output_is_none(output, "test_step")

        elif running_stage == RunningStage.EVALUATING:
            output = self.module.validation_step(*inputs, **kwargs)
            warn_if_output_is_none(output, "validation_step")

        elif running_stage == RunningStage.PREDICTING:
            output = self.module.predict(*inputs, **kwargs)
            warn_if_output_is_none(output, "predict")

        else:
            output = self.module(*inputs, **kwargs)

        return output


def warn_if_output_is_none(output: Any, method_name: str) -> None:
    """ Warns user about which method returned None. """
    if output is None:
        warning_cache.warn(f'Your {method_name} returned None. Did you forget to return an output?')


def unwrap_lightning_module(wrapped_model) -> LightningModule:
    model = wrapped_model
    if isinstance(model, (DistributedDataParallel, DataParallel)):
        model = model.module
    if isinstance(model, _LightningModuleWrapperBase):
        model = model.module
    return model
