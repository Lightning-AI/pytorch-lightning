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
from collections import OrderedDict

import torch

from pytorch_lightning.core.step_result import TrainResult, EvalResult


class TrainingStepVariations(ABC):
    """
    Houses all variations of training steps
    """

    test_step_inf_loss = float('inf')

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        self.training_step_called = True

        """Lightning calls this inside the training loop"""
        # forward pass
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self(x)

        # calculate loss
        loss_val = self.loss(y, y_hat)
        log_val = loss_val

        # alternate between tensors and scalars for "log" and "progress_bar"
        if batch_idx % 2 == 0:
            log_val = log_val.item()

        output = OrderedDict(
            {
                'loss': loss_val,
                'progress_bar': {'some_val': log_val * log_val},
                'log': {'train_some_val': log_val * log_val},
            }
        )
        return output

    def training_step__inf_loss(self, batch, batch_idx, optimizer_idx=None):
        output = self.training_step(batch, batch_idx, optimizer_idx)
        if batch_idx == self.test_step_inf_loss:
            if isinstance(output, dict):
                output['loss'] *= torch.tensor(math.inf)  # make loss infinite
            else:
                output /= 0
        return output

    def training_step_result_obj_dp(self, batch, batch_idx, optimizer_idx=None):
        # forward pass
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self(x.to(self.device))

        # calculate loss
        loss_val = self.loss(y.to(y_hat.device), y_hat)
        log_val = loss_val

        # alternate between tensors and scalars for "log" and "progress_bar"
        if batch_idx % 2 == 0:
            log_val = log_val.item()

        result = TrainResult(loss_val)
        result.log('some_val', log_val * log_val, prog_bar=True, logger=False)
        result.log('train_some_val', log_val * log_val)

        self.training_step_called = True

        return result

    def training_step_end_full_loop_result_obj_dp(self, result):
        """
        Full loop flow train step (result obj + dp)
        """
        result.minimize = result.minimize.mean()
        result.checkpoint_on = result.checkpoint_on.mean()
        result.train_step_metric = result.train_step_metric.mean()
        result.log('train_step_end_metric', 1)
        self.training_step_end_called = True
        return result

    def training_epoch_end_full_loop_result_obj_dp(self, result):
        """
        Full loop flow train step (result obj + dp)
        """
        result.log('train_epoch_end_metric', 1, on_epoch=True)
        self.training_epoch_end_called = True

        return result

    def eval_step_end_full_loop_result_obj_dp(self, result):
        """
        Full loop flow train step (result obj + dp)
        """
        eval_name = 'validation' if not self.trainer.testing else 'test'
        reduced = getattr(result, f'{eval_name}_step_metric_step').mean()
        setattr(result, f'{eval_name}_step_metric_step', reduced)

        reduced = getattr(result, f'{eval_name}_step_metric_epoch').mean()
        setattr(result, f'{eval_name}_step_metric_epoch', reduced)

        reduced = getattr(result, f'{eval_name}_step_metric').mean()
        setattr(result, f'{eval_name}_step_metric', reduced)

        result.checkpoint_on = result.checkpoint_on.mean()
        result.early_stop_on = result.early_stop_on.mean()
        result.log(f'{eval_name}_step_end_metric', torch.tensor(1).type_as(result.checkpoint_on))
        setattr(self, f'{eval_name}_step_end_called', True)

        return result

    def eval_epoch_end_full_loop_result_obj_dp(self, result):
        """
        Full loop flow train step (result obj + dp)
        """
        eval_name = 'validation' if not self.trainer.testing else 'test'
        result.log(f'{eval_name}_epoch_end_metric', torch.tensor(1).type_as(result.checkpoint_on), on_epoch=True)
        result.checkpoint_on = result.checkpoint_on.mean()
        result.early_stop_on = result.early_stop_on.mean()
        setattr(self, f'{eval_name}_epoch_end_called', True)

        # reduce the parametrized values
        reduced = getattr(result, f'{eval_name}_step_metric_step').mean()
        setattr(result, f'{eval_name}_step_metric_step', reduced)

        reduced = getattr(result, f'{eval_name}_step_metric_epoch').mean()
        setattr(result, f'{eval_name}_step_metric_epoch', reduced)

        reduced = getattr(result, f'{eval_name}_step_end_metric').mean()
        setattr(result, f'{eval_name}_step_end_metric', reduced)

        reduced = getattr(result, f'{eval_name}_step_metric').mean()
        setattr(result, f'{eval_name}_step_metric', reduced)

        return result
