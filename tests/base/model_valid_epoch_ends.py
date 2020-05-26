from abc import ABC

import torch


class ValidationEpochEndVariations(ABC):
    """
    Houses all variations of validation_epoch_end steps
    """

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs

        Args:
            outputs: list of individual outputs of each validation step
        """
        # if returned a scalar from validation_step, outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        def _mean(res, key):
            # recursive mean for multilevel dicts
            return torch.stack([x[key] if isinstance(x, dict) else _mean(x, key) for x in res]).mean()

        val_loss_mean = _mean(outputs, 'val_loss')
        val_acc_mean = _mean(outputs, 'val_acc')

        metrics_dict = {'val_loss': val_loss_mean, 'val_acc': val_acc_mean}
        results = {'progress_bar': metrics_dict, 'log': metrics_dict}
        return results

    def validation_epoch_end_multiple_dataloaders(self, outputs):
        """
        Called at the end of validation to aggregate outputs

        Args:
            outputs: list of individual outputs of each validation step
        """

        # if returned a scalar from validation_step, outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        def _mean(res, key):
            return torch.stack([x[key] for x in res]).mean()

        pbar = {}
        logs = {}
        for dl_output_list in outputs:
            output_keys = dl_output_list[0].keys()
            output_keys = [x for x in output_keys if 'val_' in x]
            for key in output_keys:
                metric_out = _mean(dl_output_list, key)
                pbar[key] = metric_out
                logs[key] = metric_out

        results = {
            'val_loss': torch.stack([v for k, v in pbar.items() if k.startswith('val_loss')]).mean(),
            'progress_bar': pbar,
            'log': logs
        }
        return results

    def validation_epoch_end__multiple_dataloaders(self, outputs):
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
