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

import logging
from abc import ABC

import torch
from torch import Tensor

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_deprecation

EPSILON = 1e-6
EPSILON_FP16 = 1e-5
log = logging.getLogger(__name__)


class TrainerTrainingTricksMixin(ABC):

    # this is just a summary on variables used in this abstract class,
    #  the proper values/initialisation should be done in child class
    lightning_module: LightningModule

    def print_nan_gradients(self) -> None:
        rank_zero_deprecation(
            "Internal: TrainerTrainingTricksMixin.print_nan_gradients is deprecated in v1.3"
            " and will be removed in v1.5."
            " Use `pytorch_lightning.utilities.nan.print_nan_gradients` instead."
        )

        model = self.lightning_module
        for param in model.parameters():
            if (param.grad is not None) and torch.isnan(param.grad.float()).any():
                log.info(param, param.grad)

    def detect_nan_tensors(self, loss: Tensor) -> None:
        rank_zero_deprecation(
            "Internal: TrainerTrainingTricksMixin.detect_nan_tensors is deprecated in v1.3"
            " and will be removed in v1.5."
            " Use `pytorch_lightning.utilities.nan.detect_nan_parameters` instead."
        )
        model = self.lightning_module

        # check if loss is nan
        if not torch.isfinite(loss).all():
            raise ValueError('The loss returned in `training_step` is nan or inf.')
        # check if a network weight is nan
        for name, param in model.named_parameters():
            if not torch.isfinite(param).all():
                self.print_nan_gradients()
                raise ValueError(
                    f'Detected nan and/or inf values in `{name}`.'
                    ' Check your forward pass for numerically unstable operations.'
                )
