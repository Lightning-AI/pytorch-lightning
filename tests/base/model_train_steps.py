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
import math
from abc import ABC

import torch


class TrainingStepVariations(ABC):
    """
    Houses all variations of training steps
    """

    test_step_inf_loss = float('inf')

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        """Lightning calls this inside the training loop"""
        self.training_step_called = True

        # forward pass
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self(x)

        # calculate loss
        loss_train = self.loss(y, y_hat)
        log_train = loss_train

        # alternate between tensors and scalars for "log" and "progress_bar"
        if batch_idx % 2 == 0:
            log_train = log_train.item()

        self.log('some_val', log_train * log_train, prog_bar=True, logger=False)
        self.log('train_some_val', log_train * log_train)
        return loss_train

    def training_step__inf_loss(self, batch, batch_idx, optimizer_idx=None):
        output = self.training_step(batch, batch_idx, optimizer_idx)
        if batch_idx == self.test_step_inf_loss:
            if isinstance(output, dict):
                output['loss'] *= torch.tensor(math.inf)  # make loss infinite
            else:
                output /= 0
        return output

    def training_step__result_obj_dp(self, batch, batch_idx, optimizer_idx=None):

        # forward pass
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self(x.to(self.device))

        # calculate loss
        loss_train = self.loss(y.to(y_hat.device), y_hat)
        log_train = loss_train

        # alternate between tensors and scalars for "log" and "progress_bar"
        if batch_idx % 2 == 0:
            log_train = log_train.item()

        self.log('some_val', log_train * log_train, prog_bar=True, logger=False)
        self.log('train_some_val', log_train * log_train)

        return loss_train
