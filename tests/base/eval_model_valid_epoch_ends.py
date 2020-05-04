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

        # return torch.stack(outputs).mean()
        val_loss_mean = _mean(outputs, 'val_loss')
        val_acc_mean = _mean(outputs, 'val_acc')
        for output in outputs:
            val_loss = self.get_output_metric(output, 'val_loss')

            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

            # reduce manually when using dp
            val_acc = self.get_output_metric(output, 'val_acc')
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_acc = torch.mean(val_acc)

            val_acc_mean += val_acc

        if outputs:  # skip zero divisions
            val_loss_mean /= len(outputs)
            val_acc_mean /= len(outputs)

        metrics_dict = {'val_loss': val_loss_mean.item(), 'val_acc': val_acc_mean.item()}
        results = {'progress_bar': metrics_dict, 'log': metrics_dict}
        return results
