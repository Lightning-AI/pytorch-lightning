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
from abc import ABC
from collections import OrderedDict

import torch


class ValidationStepVariations(ABC):
    """
    Houses all variations of validation steps
    """

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        """
        Lightning calls this inside the validation loop
        :param batch:
        :return:
        """
        self.validation_step_called = True
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self(x)

        loss_val = self.loss(y, y_hat)

        # acc
        labels_hat = torch.argmax(y_hat, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        val_acc = torch.tensor(val_acc).type_as(x)

        output = OrderedDict({
            'val_loss': loss_val,
            'val_acc': val_acc,
            'test_dic': {
                'val_loss_a': loss_val
            },
        })
        return output

    def validation_step__dp(self, batch, batch_idx, *args, **kwargs):
        self.validation_step_called = True
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self(x.to(self.device))

        y = y.to(y_hat.device)
        loss_val = self.loss(y, y_hat)

        # acc
        labels_hat = torch.argmax(y_hat, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        val_acc = torch.tensor(val_acc).type_as(x)

        self.log('val_loss', loss_val)
        self.log('val_acc', val_acc)
        return loss_val

    def validation_step__multiple_dataloaders(self, batch, batch_idx, dataloader_idx, **kwargs):
        """
        Lightning calls this inside the validation loop
        :param batch:
        :return:
        """
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self(x)

        loss_val = self.loss(y, y_hat)

        # acc
        labels_hat = torch.argmax(y_hat, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        val_acc = torch.tensor(val_acc).type_as(x)

        output = OrderedDict({
            f'val_loss_{dataloader_idx}': loss_val,
            f'val_acc_{dataloader_idx}': val_acc,
        })
        return output
