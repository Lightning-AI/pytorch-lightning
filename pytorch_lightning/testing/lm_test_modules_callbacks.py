from collections import OrderedDict
import torch


def validation_step(self, data_batch, batch_i):
    """
    Lightning calls this inside the validation loop
    :param data_batch:
    :return:
    """
    x, y = data_batch
    x = x.view(x.size(0), -1)
    y_hat = self.forward(x)

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
    if batch_i % 1 == 0:
        output = OrderedDict({
            'val_loss': loss_val,
            'val_acc': val_acc,
        })
        return output
    if batch_i % 2 == 0:
        return val_acc

    if batch_i % 3 == 0:
        output = OrderedDict({
            'val_loss': loss_val,
            'val_acc': val_acc,
            'test_dic': {'val_loss_a': loss_val}
        })
        return output


def validation_end(self, outputs):
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
    for output in outputs:
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

    val_loss_mean /= len(outputs)
    val_acc_mean /= len(outputs)

    tqdm_dic = {'val_loss': val_loss_mean.item(), 'val_acc': val_acc_mean.item()}
    return tqdm_dic


def test_end(self, outputs):
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

    test_loss_mean /= len(outputs)
    test_acc_mean /= len(outputs)

    tqdm_dic = {'test_loss': test_loss_mean.item(), 'test_acc': test_acc_mean.item()}
    return tqdm_dic


def test_step(self, data_batch, batch_i):
    """
    Lightning calls this inside the validation loop
    :param data_batch:
    :return:
    """
    x, y = data_batch
    x = x.view(x.size(0), -1)
    y_hat = self.forward(x)

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
    if batch_i % 1 == 0:
        output = OrderedDict({
            'test_loss': loss_test,
            'test_acc': test_acc,
        })
        return output
    if batch_i % 2 == 0:
        return test_acc

    if batch_i % 3 == 0:
        output = OrderedDict({
            'test_loss': loss_test,
            'test_acc': test_acc,
            'test_dic': {'test_loss_a': loss_test}
        })
    return output