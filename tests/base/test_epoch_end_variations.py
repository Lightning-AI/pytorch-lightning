import torch


class TestEpochEndVariationsMixin:
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


def _get_output_metric(output, name):
    if isinstance(output, dict):
        val = output[name]
    else:  # if it is 2level deep -> per dataloader and per batch
        val = sum(out[name] for out in output) / len(output)
    return val
