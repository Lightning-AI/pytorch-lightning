import random
from abc import ABC
from collections import OrderedDict

import torch

from pytorch_lightning import EvalResult


class TestStepVariations(ABC):
    """
    Houses all variations of test steps
    """

    def test_step(self, batch, batch_idx, *args, **kwargs):
        """
        Default, baseline test_step
        :param batch:
        :return:
        """
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self(x)

        loss_test = self.loss(y, y_hat)

        # acc
        labels_hat = torch.argmax(y_hat, dim=1)
        test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        test_acc = torch.tensor(test_acc)

        test_acc = test_acc.type_as(x)

        # alternate possible outputs to test
        if batch_idx % 1 == 0:
            output = OrderedDict({
                'test_loss': loss_test,
                'test_acc': test_acc,
            })
            return output
        if batch_idx % 2 == 0:
            return test_acc

        if batch_idx % 3 == 0:
            output = OrderedDict({
                'test_loss': loss_test,
                'test_acc': test_acc,
                'test_dic': {'test_loss_a': loss_test}
            })
            return output

    def test_step__multiple_dataloaders(self, batch, batch_idx, dataloader_idx, **kwargs):
        """
        Default, baseline test_step
        :param batch:
        :return:
        """
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self(x)

        loss_test = self.loss(y, y_hat)

        # acc
        labels_hat = torch.argmax(y_hat, dim=1)
        test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        test_acc = torch.tensor(test_acc)

        test_acc = test_acc.type_as(x)

        # alternate possible outputs to test
        if batch_idx % 1 == 0:
            output = OrderedDict({
                'test_loss': loss_test,
                'test_acc': test_acc,
            })
            return output
        if batch_idx % 2 == 0:
            return test_acc

        if batch_idx % 3 == 0:
            output = OrderedDict({
                'test_loss': loss_test,
                'test_acc': test_acc,
                'test_dic': {'test_loss_a': loss_test}
            })
            return output
        if batch_idx % 5 == 0:
            output = OrderedDict({
                f'test_loss_{dataloader_idx}': loss_test,
                f'test_acc_{dataloader_idx}': test_acc,
            })
            return output

    def test_step__empty(self, batch, batch_idx, *args, **kwargs):
        return {}


    def test_step_result_preds(self, batch, batch_idx, optimizer_idx=None):
        """Lightning calls this inside the training loop"""
        """
        Default, baseline test_step
        :param batch:
        :return:
        """
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self(x)

        loss_test = self.loss(y, y_hat)

        # acc
        labels_hat = torch.argmax(y_hat, dim=1)
        test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        test_acc = torch.tensor(test_acc)

        test_acc = test_acc.type_as(x)

        # Do regular EvalResult Logging
        result = EvalResult(checkpoint_on=loss_test)
        result.log('test_loss', loss_test)
        result.log('test_acc', test_acc)

        #lst_of_str = [random.choice(['dog', 'cat']) for i in range(batch_size)]
        # int_outputs = [random.randint(500, 1000) for i in range(batch_size)]
        #nested_lst = [[x] for x in int_outputs]
        #lst_of_dicts = [{k: v} for k, v in zip(lst_of_str, int_outputs)]

        # This is passed in from pytest via parameterization
        option = getattr(self, 'test_option', 0)

        lazy_ids = torch.arange(batch_idx * self.batch_size, (batch_idx + 1) * x.size(0))

        # Base
        if option == 0:
            result.write('idxs', lazy_ids)
            result.write('preds', labels_hat)

        # Check mismatching tensor len
        elif option == 1:
            result.write('idxs', torch.cat((lazy_ids, lazy_ids)))
            result.write('preds', labels_hat)

        return result
