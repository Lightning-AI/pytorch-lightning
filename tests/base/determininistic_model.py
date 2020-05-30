import torch
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.core.step_result import TrainResult, EvalResult


class DeterministicModel(LightningModule):

    def __init__(self, weights=None):
        super().__init__()
        if weights is None:
            weights = torch.tensor([
                [1, 3, 5],
                [7, 11, 13]
            ])
        self.l1 = weights

    def forward(self, x):
        return self.l1.mm(x)

    def base_eval_result(self, acc, x):
        result = TrainResult(
            minimize=acc,
            early_stop_on=torch.tensor(1.4).type_as(x),
            checkpoint_on=torch.tensor(1.5).type_as(x)
        )

        result.log('log_acc1', torch.tensor(12).type_as(x))
        result.log('log_acc1', torch.tensor(7).type_as(x))
        result.to_pbar('pbar_acc1', torch.tensor(17).type_as(x))
        result.to_pbar('pbar_acc2', torch.tensor(19).type_as(x))
        return result

    def training_step_only(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        acc = torch.all(y_hat, y)

        result = self.base_eval_result(acc, x)
        return result

    def training_step_with_batch_end(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        acc = torch.all(y_hat, y)

        result = self.base_eval_result(acc, x)
        result.to_batch_end('to_batch_end_1', torch.tensor([-1, -2, -3]).type_as(x))

        return result

    def training_step_with_epoch_end(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        acc = torch.all(y_hat, y)

        result = self.base_eval_result(acc, x)
        result.to_epoch_end('to_epoch_end_1', torch.tensor([-3, -2, -3]).type_as(x))

        return result

    def training_step_with_batch_and_epoch_end(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        acc = torch.all(y_hat, y)

        result = self.base_eval_result(acc, x)
        result.to_batch_end('to_batch_end_1', torch.tensor([-1, -2, -3]).type_as(x))
        result.to_epoch_end('to_epoch_end_1', torch.tensor([-3, -2, -3]).type_as(x))

        return result

    def training_step_dict_return(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        acc = torch.all(y_hat, y)

        logs = {'log_acc1': acc * 12, 'log_acc2': acc * 7}
        pbar = {'pbar_acc1': acc * 17, 'pbar_acc2': acc * 19}
        return {'loss': acc, 'log': logs, 'progress_bar': pbar}

    def training_step_end(self, outputs):
        if self.use_dp or self.use_ddp2:
            pass
        else:
            # only saw 3 batches
            assert len(outputs) == 3
