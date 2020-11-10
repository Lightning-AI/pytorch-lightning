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

from torch import optim


class ConfigureOptimizersPool(ABC):

    def configure_optimizers(self):
        """
        return whatever optimizers we want here.
        :return: list of optimizers
        """
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def configure_optimizers__empty(self):
        return None

    def configure_optimizers__lbfgs(self):
        """
        return whatever optimizers we want here.
        :return: list of optimizers
        """
        optimizer = optim.LBFGS(self.parameters(), lr=self.learning_rate)
        return optimizer

    def configure_optimizers__adagrad(self):
        optimizer = optim.Adagrad(self.parameters(), lr=self.learning_rate)
        return optimizer

    def configure_optimizers__multiple_optimizers(self):
        """
        return whatever optimizers we want here.
        :return: list of optimizers
        """
        # try no scheduler for this model (testing purposes)
        optimizer1 = optim.Adam(self.parameters(), lr=self.learning_rate)
        optimizer2 = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer1, optimizer2

    def configure_optimizers__multiple_optimizers_frequency(self):
        optimizer1 = optim.Adam(self.parameters(), lr=self.learning_rate)
        optimizer2 = optim.Adam(self.parameters(), lr=self.learning_rate)
        return [
            {'optimizer': optimizer1, 'frequency': 1},
            {'optimizer': optimizer2, 'frequency': 5}
        ]

    def configure_optimizers__single_scheduler(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1)
        return [optimizer], [lr_scheduler]

    def configure_optimizers__onecycle_scheduler(self):
        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                                     max_lr=self.learning_rate,
                                                     total_steps=10_000)
        return [optimizer], [lr_scheduler]

    def configure_optimizers__multiple_schedulers(self):
        optimizer1 = optim.Adam(self.parameters(), lr=self.learning_rate)
        optimizer2 = optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler1 = optim.lr_scheduler.StepLR(optimizer1, 1, gamma=0.1)
        lr_scheduler2 = optim.lr_scheduler.StepLR(optimizer2, 1, gamma=0.1)

        return [optimizer1, optimizer2], [lr_scheduler1, lr_scheduler2]

    def configure_optimizers__mixed_scheduling(self):
        optimizer1 = optim.Adam(self.parameters(), lr=self.learning_rate)
        optimizer2 = optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler1 = optim.lr_scheduler.StepLR(optimizer1, 4, gamma=0.1)
        lr_scheduler2 = optim.lr_scheduler.StepLR(optimizer2, 1, gamma=0.1)

        return [optimizer1, optimizer2], \
            [{'scheduler': lr_scheduler1, 'interval': 'step'}, lr_scheduler2]

    def configure_optimizers__param_groups(self):
        param_groups = [
            {'params': list(self.parameters())[:2], 'lr': self.learning_rate * 0.1},
            {'params': list(self.parameters())[2:], 'lr': self.learning_rate}
        ]

        optimizer = optim.Adam(param_groups)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1)
        return [optimizer], [lr_scheduler]

    def configure_optimizers__lr_from_hparams(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
