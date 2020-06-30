from abc import ABC

import torch


class TestEpochEndVariations(ABC):

    def test_epoch_end(self, outputs):
        """
        Called at the end of test epoch to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        """
        # if returned a scalar from test_step, outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        # return torch.stack(outputs).mean()
        test_loss_mean = 0
        test_acc_mean = 0
        for output in outputs:
            test_loss = self.get_output_metric(output, 'test_loss')

            # reduce manually when using dp
            if self.trainer.use_dp:
                test_loss = torch.mean(test_loss)
            test_loss_mean += test_loss

            # reduce manually when using dp
            test_acc = self.get_output_metric(output, 'test_acc')
            if self.trainer.use_dp:
                test_acc = torch.mean(test_acc)

            test_acc_mean += test_acc

        test_loss_mean /= len(outputs)
        test_acc_mean /= len(outputs)

        metrics_dict = {'test_loss': test_loss_mean, 'test_acc': test_acc_mean}
        result = {'progress_bar': metrics_dict, 'log': metrics_dict}
        return result

    def test_epoch_end__multiple_dataloaders(self, outputs):
        """
        Called at the end of test epoch to aggregate outputs
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

        tqdm_dict = {'test_loss': test_loss_mean, 'test_acc': test_acc_mean}
        result = {'progress_bar': tqdm_dict}
        return result
