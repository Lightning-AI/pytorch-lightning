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

from abc import ABC, abstractmethod
import logging

import torch
from torch import Tensor

from pytorch_lightning.core.lightning import LightningModule

EPSILON = 1e-6
EPSILON_FP16 = 1e-5
log = logging.getLogger(__name__)


class TrainerTrainingTricksMixin(ABC):

    # this is just a summary on variables used in this abstract class,
    #  the proper values/initialisation should be done in child class
    default_root_dir: str
    progress_bar_callback: ...
    on_gpu: bool

    @abstractmethod
    def get_model(self) -> LightningModule:
        """Warning: this is just empty shell for code implemented in other class."""

    def print_nan_gradients(self) -> None:
        model = self.get_model()
        for param in model.parameters():
            if (param.grad is not None) and torch.isnan(param.grad.float()).any():
                log.info(param, param.grad)

    def detect_nan_tensors(self, loss: Tensor) -> None:
        model = self.get_model()

        # check if loss is nan
        if not torch.isfinite(loss).all():
            raise ValueError(
                'The loss returned in `training_step` is nan or inf.'
            )
        # check if a network weight is nan
        for name, param in model.named_parameters():
            if not torch.isfinite(param).all():
                self.print_nan_gradients()
                raise ValueError(
                    f'Detected nan and/or inf values in `{name}`.'
                    ' Check your forward pass for numerically unstable operations.'
                )
