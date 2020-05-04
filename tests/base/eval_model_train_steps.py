import math
from abc import ABC
from collections import OrderedDict

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
        if self.trainer.batch_idx % 1 == 0:
            output = OrderedDict({
                'loss': loss_val,
                'progress_bar': {'some_val': loss_val * loss_val},
                'log': {'train_some_val': loss_val * loss_val},
            })
            return output

        if self.trainer.batch_idx % 2 == 0:
            return loss_val

    def training_step__inf_loss(self, batch, batch_idx, optimizer_idx=None):
        output = self.training_step(batch, batch_idx, optimizer_idx)
        if batch_idx == self.test_step_inf_loss:
            if isinstance(output, dict):
                output['loss'] *= torch.tensor(math.inf)  # make loss infinite
            else:
                output /= 0
        return output
