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
import random
from abc import ABC
from collections import OrderedDict

import torch

from pytorch_lightning.core.step_result import EvalResult


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
        self.test_step_called = True

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
            output = OrderedDict({'test_loss': loss_test, 'test_acc': test_acc})
            return output
        if batch_idx % 2 == 0:
            return test_acc

        if batch_idx % 3 == 0:
            output = OrderedDict({'test_loss': loss_test,
                                  'test_acc': test_acc,
                                  'test_dic': {'test_loss_a': loss_test}})
            return output

    def test_step_result_obj(self, batch, batch_idx, *args, **kwargs):
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

        result = EvalResult()
        # alternate possible outputs to test
        if batch_idx % 1 == 0:
            result.log_dict({'test_loss': loss_test, 'test_acc': test_acc})
            return result
        if batch_idx % 2 == 0:
            return test_acc

        if batch_idx % 3 == 0:
            result.log_dict({'test_loss': loss_test, 'test_acc': test_acc})
            result.test_dic = {'test_loss_a': loss_test}
            return result

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
            output = OrderedDict({'test_loss': loss_test, 'test_acc': test_acc})
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
            output = OrderedDict({f'test_loss_{dataloader_idx}': loss_test, f'test_acc_{dataloader_idx}': test_acc})
            return output

    def test_step__empty(self, batch, batch_idx, *args, **kwargs):
        return {}

    def test_step_result_preds(self, batch, batch_idx, optimizer_idx=None):
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

        batch_size = x.size(0)
        lst_of_str = [random.choice(['dog', 'cat']) for i in range(batch_size)]
        lst_of_int = [random.randint(500, 1000) for i in range(batch_size)]
        lst_of_lst = [[x] for x in lst_of_int]
        lst_of_dict = [{k: v} for k, v in zip(lst_of_str, lst_of_int)]

        # This is passed in from pytest via parameterization
        option = getattr(self, 'test_option', 0)
        prediction_file = getattr(self, 'prediction_file', 'predictions.pt')

        lazy_ids = torch.arange(batch_idx * self.batch_size, batch_idx * self.batch_size + x.size(0))

        # Base
        if option == 0:
            self.write_prediction('idxs', lazy_ids, prediction_file)
            self.write_prediction('preds', labels_hat, prediction_file)

        # Check mismatching tensor len
        elif option == 1:
            self.write_prediction('idxs', torch.cat((lazy_ids, lazy_ids)), prediction_file)
            self.write_prediction('preds', labels_hat, prediction_file)

        # write multi-dimension
        elif option == 2:
            self.write_prediction('idxs', lazy_ids, prediction_file)
            self.write_prediction('preds', labels_hat, prediction_file)
            self.write_prediction('x', x, prediction_file)

        # write str list
        elif option == 3:
            self.write_prediction('idxs', lazy_ids, prediction_file)
            self.write_prediction('vals', lst_of_str, prediction_file)

        # write int list
        elif option == 4:
            self.write_prediction('idxs', lazy_ids, prediction_file)
            self.write_prediction('vals', lst_of_int, prediction_file)

        # write nested list
        elif option == 5:
            self.write_prediction('idxs', lazy_ids, prediction_file)
            self.write_prediction('vals', lst_of_lst, prediction_file)

        # write dict list
        elif option == 6:
            self.write_prediction('idxs', lazy_ids, prediction_file)
            self.write_prediction('vals', lst_of_dict, prediction_file)

        elif option == 7:
            self.write_prediction_dict({'idxs': lazy_ids, 'preds': labels_hat}, prediction_file)

        return result
