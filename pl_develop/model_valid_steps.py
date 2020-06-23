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
            'test_dic': {'val_loss_a': loss_val}
        })
        return output

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
