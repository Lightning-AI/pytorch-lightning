import math
from abc import ABC
from collections import OrderedDict
from pytorch_lightning import TrainResult

import torch


class TrainingStepVariations(ABC):
    """
    Houses all variations of training steps
    """
    test_step_inf_loss = float('inf')

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        """Lightning calls this inside the training loop"""
        # forward pass
        x, y = batch
        x = x.view(x.size(0), -1)

        y_hat = self(x)

        # calculate loss
        loss_val = self.loss(y, y_hat)

        # alternate possible outputs to test
        output = OrderedDict({
            'loss': loss_val,
            'progress_bar': {'some_val': loss_val * loss_val},
            'log': {'train_some_val': loss_val * loss_val},
        })
        return output

    def training_step__inf_loss(self, batch, batch_idx, optimizer_idx=None):
        output = self.training_step(batch, batch_idx, optimizer_idx)
        if batch_idx == self.test_step_inf_loss:
            if isinstance(output, dict):
                output['loss'] *= torch.tensor(math.inf)  # make loss infinite
            else:
                output /= 0
        return output

    def training_step_full_loop_result_obj(self, batch, batch_idx):
        """
        Full loop flow train step
        """
        x, y = batch
        x = x.view(x.size(0), -1)
        print(self.device, self.c_d1.weight.device, x.device)
        y_hat = self(x)
        loss_val = self.loss(y.type_as(y_hat).float(), y_hat.long())
        result = TrainResult(minimize=loss_val)
        result.log('train_step_acc1', loss_val + 1)
        self.training_step_called = True
        return result

    def training_step_end_full_loop_result_obj_dp(self, result):
        """
        Full loop flow train step
        """
        result.minimize = result.minimize.mean()
        result.checkpoint_on = result.checkpoint_on.mean()
        result.train_step_acc1 = result.train_step_acc1.mean()
        result.log('train_step_end_acc1', 1)
        self.training_step_end_called = True
        return result

    def training_epoch_end_full_loop_result_obj(self, result):
        """
        Full loop flow train step
        """
        result.log('train_epoch_end_acc1', 1)
        self.training_epoch_end_called = True
        return result
