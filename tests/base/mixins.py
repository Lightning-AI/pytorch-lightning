from collections import OrderedDict

import torch
from torch import optim


class LightValidationStepMixin:
    """
    Add val_dataloader and validation_step methods for the case
    when val_dataloader returns a single dataloader
    """

    def val_dataloader(self):
        return self._dataloader(train=False)

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        """Lightning calls this inside the validation loop."""
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self(x)

        loss_val = self.loss(y, y_hat)

        # acc
        labels_hat = torch.argmax(y_hat, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        val_acc = torch.tensor(val_acc)

        if self.on_gpu:
            val_acc = val_acc.cuda(loss_val.device.index)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp:
            loss_val = loss_val.unsqueeze(0)
            val_acc = val_acc.unsqueeze(0)

        # alternate possible outputs to test
        if batch_idx % 1 == 0:
            output = OrderedDict({
                'val_loss': loss_val,
                'val_acc': val_acc,
            })
            return output
        if batch_idx % 2 == 0:
            return val_acc

        if batch_idx % 3 == 0:
            output = OrderedDict({
                'val_loss': loss_val,
                'val_acc': val_acc,
                'test_dic': {'val_loss_a': loss_val}
            })
            return output


class LightValidationMixin(LightValidationStepMixin):
    """
    Add val_dataloader, validation_step, and validation_end methods for the case
    when val_dataloader returns a single dataloader
    """

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs

        Args:
            outputs: list of individual outputs of each validation step
        """
        # if returned a scalar from validation_step, outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        # return torch.stack(outputs).mean()
        val_loss_mean = 0
        val_acc_mean = 0
        for output in outputs:
            val_loss = _get_output_metric(output, 'val_loss')

            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

            # reduce manually when using dp
            val_acc = _get_output_metric(output, 'val_acc')
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_acc = torch.mean(val_acc)

            val_acc_mean += val_acc

        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)

        metrics_dict = {'val_loss': val_loss_mean.item(), 'val_acc': val_acc_mean.item()}
        results = {'progress_bar': metrics_dict, 'log': metrics_dict}
        return results


class LightValidationStepMultipleDataloadersMixin:
    """
    Add val_dataloader and validation_step methods for the case
    when val_dataloader returns multiple dataloaders
    """

    def val_dataloader(self):
        return [self._dataloader(train=False), self._dataloader(train=False)]

    def validation_step(self, batch, batch_idx, dataloader_idx, **kwargs):
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
        val_acc = torch.tensor(val_acc)

        if self.on_gpu:
            val_acc = val_acc.cuda(loss_val.device.index)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp:
            loss_val = loss_val.unsqueeze(0)
            val_acc = val_acc.unsqueeze(0)

        # alternate possible outputs to test
        if batch_idx % 1 == 0:
            output = OrderedDict({
                'val_loss': loss_val,
                'val_acc': val_acc,
            })
            return output
        if batch_idx % 2 == 0:
            return val_acc

        if batch_idx % 3 == 0:
            output = OrderedDict({
                'val_loss': loss_val,
                'val_acc': val_acc,
                'test_dic': {'val_loss_a': loss_val}
            })
            return output
        if batch_idx % 5 == 0:
            output = OrderedDict({
                f'val_loss_{dataloader_idx}': loss_val,
                f'val_acc_{dataloader_idx}': val_acc,
            })
            return output


class LightValidationMultipleDataloadersMixin(LightValidationStepMultipleDataloadersMixin):
    """
    Add val_dataloader, validation_step, and validation_end methods for the case
    when val_dataloader returns multiple dataloaders
    """

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        """
        # if returned a scalar from validation_step, outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        # return torch.stack(outputs).mean()
        val_loss_mean = 0
        val_acc_mean = 0
        i = 0
        for dl_output in outputs:
            for output in dl_output:
                val_loss = output['val_loss']

                # reduce manually when using dp
                if self.trainer.use_dp:
                    val_loss = torch.mean(val_loss)
                val_loss_mean += val_loss

                # reduce manually when using dp
                val_acc = output['val_acc']
                if self.trainer.use_dp:
                    val_acc = torch.mean(val_acc)

                val_acc_mean += val_acc
                i += 1

        val_loss_mean /= i
        val_acc_mean /= i

        tqdm_dict = {'val_loss': val_loss_mean.item(), 'val_acc': val_acc_mean.item()}
        result = {'progress_bar': tqdm_dict}
        return result


class LightTrainDataloader:
    """Simple train dataloader."""

    def train_dataloader(self):
        return self._dataloader(train=True)


class LightValidationDataloader:
    """Simple validation dataloader."""

    def val_dataloader(self):
        return self._dataloader(train=False)


class LightTestDataloader:
    """Simple test dataloader."""

    def test_dataloader(self):
        return self._dataloader(train=False)


class CustomInfDataloader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iter = iter(dataloader)
        self.count = 0

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count >= 50:
            raise StopIteration
        self.count = self.count + 1
        try:
            return next(self.iter)
        except StopIteration:
            self.iter = iter(self.dataloader)
            return next(self.iter)


class LightInfTrainDataloader:
    """Simple test dataloader."""

    def train_dataloader(self):
        return CustomInfDataloader(self._dataloader(train=True))


class LightInfValDataloader:
    """Simple test dataloader."""

    def val_dataloader(self):
        return CustomInfDataloader(self._dataloader(train=False))


class LightInfTestDataloader:
    """Simple test dataloader."""

    def test_dataloader(self):
        return CustomInfDataloader(self._dataloader(train=False))


class LightZeroLenDataloader:
    """ Simple dataloader that has zero length. """

    def train_dataloader(self):
        dataloader = self._dataloader(train=True)
        dataloader.dataset.data = dataloader.dataset.data[:0]
        dataloader.dataset.targets = dataloader.dataset.targets[:0]
        return dataloader


class LightEmptyTestStep:
    """Empty test step."""

    def test_step(self, *args, **kwargs):
        return dict()


class LightTestStepMixin(LightTestDataloader):
    """Test step mixin."""

    def test_step(self, batch, batch_idx, *args, **kwargs):
        """
        Lightning calls this inside the validation loop
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

        if self.on_gpu:
            test_acc = test_acc.cuda(loss_test.device.index)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp:
            loss_test = loss_test.unsqueeze(0)
            test_acc = test_acc.unsqueeze(0)

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


class LightTestMixin(LightTestStepMixin):
    """Ritch test mixin."""

    def test_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        """
        # if returned a scalar from test_step, outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        # return torch.stack(outputs).mean()
        test_loss_mean = 0
        test_acc_mean = 0
        for output in outputs:
            test_loss = _get_output_metric(output, 'test_loss')

            # reduce manually when using dp
            if self.trainer.use_dp:
                test_loss = torch.mean(test_loss)
            test_loss_mean += test_loss

            # reduce manually when using dp
            test_acc = _get_output_metric(output, 'test_acc')
            if self.trainer.use_dp:
                test_acc = torch.mean(test_acc)

            test_acc_mean += test_acc

        test_loss_mean /= len(outputs)
        test_acc_mean /= len(outputs)

        metrics_dict = {'test_loss': test_loss_mean.item(), 'test_acc': test_acc_mean.item()}
        result = {'progress_bar': metrics_dict, 'log': metrics_dict}
        return result


class LightTestStepMultipleDataloadersMixin:
    """Test step multiple dataloaders mixin."""

    def test_dataloader(self):
        return [self._dataloader(train=False), self._dataloader(train=False)]

    def test_step(self, batch, batch_idx, dataloader_idx, **kwargs):
        """
        Lightning calls this inside the validation loop
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

        if self.on_gpu:
            test_acc = test_acc.cuda(loss_test.device.index)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp:
            loss_test = loss_test.unsqueeze(0)
            test_acc = test_acc.unsqueeze(0)

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


class LightTestFitSingleTestDataloadersMixin:
    """Test fit single test dataloaders mixin."""

    def test_dataloader(self):
        return self._dataloader(train=False)

    def test_step(self, batch, batch_idx, *args, **kwargs):
        """
        Lightning calls this inside the validation loop
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

        if self.on_gpu:
            test_acc = test_acc.cuda(loss_test.device.index)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp:
            loss_test = loss_test.unsqueeze(0)
            test_acc = test_acc.unsqueeze(0)

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


class LightTestFitMultipleTestDataloadersMixin:
    """Test fit multiple test dataloaders mixin."""

    def test_step(self, batch, batch_idx, dataloader_idx, **kwargs):
        """
        Lightning calls this inside the validation loop
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

        if self.on_gpu:
            test_acc = test_acc.cuda(loss_test.device.index)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp:
            loss_test = loss_test.unsqueeze(0)
            test_acc = test_acc.unsqueeze(0)

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


class LightValStepFitSingleDataloaderMixin:

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
        val_acc = torch.tensor(val_acc)

        if self.on_gpu:
            val_acc = val_acc.cuda(loss_val.device.index)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp:
            loss_val = loss_val.unsqueeze(0)
            val_acc = val_acc.unsqueeze(0)

        # alternate possible outputs to test
        if batch_idx % 1 == 0:
            output = OrderedDict({
                'val_loss': loss_val,
                'val_acc': val_acc,
            })
            return output
        if batch_idx % 2 == 0:
            return val_acc

        if batch_idx % 3 == 0:
            output = OrderedDict({
                'val_loss': loss_val,
                'val_acc': val_acc,
                'test_dic': {'val_loss_a': loss_val}
            })
            return output


class LightValStepFitMultipleDataloadersMixin:

    def validation_step(self, batch, batch_idx, dataloader_idx, **kwargs):
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
        val_acc = torch.tensor(val_acc)

        if self.on_gpu:
            val_acc = val_acc.cuda(loss_val.device.index)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp:
            loss_val = loss_val.unsqueeze(0)
            val_acc = val_acc.unsqueeze(0)

        # alternate possible outputs to test
        if batch_idx % 1 == 0:
            output = OrderedDict({
                'val_loss': loss_val,
                'val_acc': val_acc,
            })
            return output
        if batch_idx % 2 == 0:
            return val_acc

        if batch_idx % 3 == 0:
            output = OrderedDict({
                'val_loss': loss_val,
                'val_acc': val_acc,
                'test_dic': {'val_loss_a': loss_val}
            })
            return output
        if batch_idx % 5 == 0:
            output = OrderedDict({
                f'val_loss_{dataloader_idx}': loss_val,
                f'val_acc_{dataloader_idx}': val_acc,
            })
            return output


class LightTestMultipleDataloadersMixin(LightTestStepMultipleDataloadersMixin):

    def test_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        """
        # if returned a scalar from test_step, outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        # return torch.stack(outputs).mean()
        test_loss_mean = 0
        test_acc_mean = 0
        i = 0
        for dl_output in outputs:
            for output in dl_output:
                test_loss = output['test_loss']

                # reduce manually when using dp
                if self.trainer.use_dp:
                    test_loss = torch.mean(test_loss)
                test_loss_mean += test_loss

                # reduce manually when using dp
                test_acc = output['test_acc']
                if self.trainer.use_dp:
                    test_acc = torch.mean(test_acc)

                test_acc_mean += test_acc
                i += 1

        test_loss_mean /= i
        test_acc_mean /= i

        tqdm_dict = {'test_loss': test_loss_mean.item(), 'test_acc': test_acc_mean.item()}
        result = {'progress_bar': tqdm_dict}
        return result


class LightTestOptimizerWithSchedulingMixin:
    def configure_optimizers(self):
        if self.hparams.optimizer_name == 'lbfgs':
            optimizer = optim.LBFGS(self.parameters(), lr=self.hparams.learning_rate)
        else:
            optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1)
        return [optimizer], [lr_scheduler]


class LightTestMultipleOptimizersWithSchedulingMixin:
    def configure_optimizers(self):
        if self.hparams.optimizer_name == 'lbfgs':
            optimizer1 = optim.LBFGS(self.parameters(), lr=self.hparams.learning_rate)
            optimizer2 = optim.LBFGS(self.parameters(), lr=self.hparams.learning_rate)
        else:
            optimizer1 = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
            optimizer2 = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        lr_scheduler1 = optim.lr_scheduler.StepLR(optimizer1, 1, gamma=0.1)
        lr_scheduler2 = optim.lr_scheduler.StepLR(optimizer2, 1, gamma=0.1)

        return [optimizer1, optimizer2], [lr_scheduler1, lr_scheduler2]


class LightTestOptimizersWithMixedSchedulingMixin:
    def configure_optimizers(self):
        if self.hparams.optimizer_name == 'lbfgs':
            optimizer1 = optim.LBFGS(self.parameters(), lr=self.hparams.learning_rate)
            optimizer2 = optim.LBFGS(self.parameters(), lr=self.hparams.learning_rate)
        else:
            optimizer1 = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
            optimizer2 = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        lr_scheduler1 = optim.lr_scheduler.StepLR(optimizer1, 4, gamma=0.1)
        lr_scheduler2 = optim.lr_scheduler.StepLR(optimizer2, 1, gamma=0.1)

        return [optimizer1, optimizer2], \
            [{'scheduler': lr_scheduler1, 'interval': 'step'}, lr_scheduler2]


class LightTestReduceLROnPlateauMixin:
    def configure_optimizers(self):
        if self.hparams.optimizer_name == 'lbfgs':
            optimizer = optim.LBFGS(self.parameters(), lr=self.hparams.learning_rate)
        else:
            optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return [optimizer], [lr_scheduler]


class LightTestNoneOptimizerMixin:
    def configure_optimizers(self):
        return None


def _get_output_metric(output, name):
    if isinstance(output, dict):
        val = output[name]
    else:  # if it is 2level deep -> per dataloader and per batch
        val = sum(out[name] for out in output) / len(output)
    return val
