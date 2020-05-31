import torch
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.core.step_result import TrainResult, EvalResult
from torch.utils.data import Dataset, DataLoader
import numpy as np


class DeterministicModel(LightningModule):

    def __init__(self, weights=None):
        super().__init__()
        if weights is None:
            weights = torch.tensor([
                [4, 3, 5],
                [10, 11, 13]
            ]).float()
        self.l1 = torch.nn.Parameter(weights, requires_grad=True)

    def forward(self, x):
        return self.l1.mm(x.float().t())

    def base_train_result(self, acc):
        x = acc
        result = TrainResult(
            minimize=acc,
            early_stop_on=torch.tensor(1.4).type_as(x),
            checkpoint_on=torch.tensor(1.5).type_as(x)
        )

        result.log('log_acc1', torch.tensor(12).type_as(x))
        result.log('log_acc2', torch.tensor(7).type_as(x))
        result.to_pbar('pbar_acc1', torch.tensor(17).type_as(x))
        result.to_pbar('pbar_acc2', torch.tensor(19).type_as(x))

        # make sure minimize is the only thing with a graph
        self.assert_graph_count(result, 1)
        return result

    def base_eval_result(self, acc):
        x = acc
        result = EvalResult(
            early_stop_on=torch.tensor(1.4).type_as(x),
            checkpoint_on=torch.tensor(1.5).type_as(x)
        )
        result.log('log_acc1', torch.tensor(12).type_as(x))
        result.log('log_acc2', torch.tensor(7).type_as(x))
        result.to_pbar('pbar_acc1', torch.tensor(17).type_as(x))
        result.to_pbar('pbar_acc2', torch.tensor(19).type_as(x))
        return result

    def step(self, batch, batch_idx):
        x = batch
        y_hat = self(x)

        assert torch.all(y_hat[0, :] == 15.0)
        assert torch.all(y_hat[1, :] == 42.0)
        out = y_hat.sum()
        assert out == (42.0*3) + (15.0*3)

        return out

    def assert_graph_count(self, result, count=1):
        counts = self.count_num_graphs(result)
        assert counts == count

    def count_num_graphs(self, result: TrainResult, num_graphs=0):
        for k, v in result.items():
            if isinstance(v, torch.Tensor) and v.grad_fn is not None:
                num_graphs += 1
            if isinstance(v, dict):
                num_graphs += self.count_num_graphs(v)

        return num_graphs

    def training_step_only(self, batch, batch_idx):
        acc = self.step(batch, batch_idx)

        result = self.base_train_result(acc)
        return result

    def training_step_with_batch_end(self, batch, batch_idx):
        acc = self.step(batch, batch_idx)

        result = self.base_train_result(acc)
        result.pass_to_batch_end('to_batch_end_1', torch.tensor([-1, -2, -3]).type_as(acc))

        return result

    def training_step_with_epoch_end(self, batch, batch_idx):
        acc = self.step(batch, batch_idx)

        result = self.base_train_result(acc)
        result.pass_to_epoch_end('to_epoch_end_1', torch.tensor([-3, -2, -3]).type_as(acc))

        return result

    def training_step_with_batch_and_epoch_end(self, batch, batch_idx):
        acc = self.step(batch, batch_idx)

        result = self.base_train_result(acc)
        result.pass_to_batch_end('to_batch_end_1', torch.tensor([-1, -2, -3]).type_as(acc))
        result.pass_to_epoch_end('to_epoch_end_1', torch.tensor([-3, -2, -3]).type_as(acc))

        return result

    def training_step_dict_return(self, batch, batch_idx):
        acc = self.step(batch, batch_idx)

        logs = {'log_acc1': torch.tensor(12).type_as(acc), 'log_acc2': torch.tensor(7).type_as(acc)}
        pbar = {'pbar_acc1': torch.tensor(17).type_as(acc), 'pbar_acc2': torch.tensor(19).type_as(acc)}
        return {'loss': acc, 'log': logs, 'progress_bar': pbar}

    def training_step_end_basic(self, outputs):
        # make sure only the expected keys are here
        keys = set(outputs.keys())
        assert keys == {'to_batch_end_1', 'minimize'}

        result = TrainResult()
        result.pass_to_epoch_end('from_train_step_end', torch.tensor(19))
        return result

    def training_epoch_end_basic(self, outputs):
        if self.use_dp or self.use_ddp2:
            pass
        else:
            # only saw 3 batches
            assert len(outputs) == 3
            for batch_out in outputs:
                assert len(batch_out.keys()) == 2
                keys = ['to_batch_end_1', 'to_batch_end_2']
                for key in keys:
                    assert key in batch_out

    def validation_step_only(self, batch, batch_idx):
        acc = self.step(batch, batch_idx)

        result = self.base_eval_result(acc)

        return result

    def validation_step_with_batch_end(self, batch, batch_idx):
        acc = self.step(batch, batch_idx)

        result = self.base_eval_result(acc)
        result.pass_to_batch_end('to_batch_end_1', torch.tensor([-1, -2, -3]).type_as(acc))

        return result

    def validation_step_with_epoch_end(self, batch, batch_idx):
        acc = self.step(batch, batch_idx)

        result = self.base_eval_result(acc)
        result.pass_to_epoch_end('to_epoch_end_1', torch.tensor([-3, -2, -3]).type_as(acc))

        return result

    def validation_step_with_batch_and_epoch_end(self, batch, batch_idx):
        acc = self.step(batch, batch_idx)

        result = self.base_eval_result(acc)
        result.pass_to_batch_end('to_batch_end_1', torch.tensor([-1, -2, -3]).type_as(acc))
        result.pass_to_epoch_end('to_epoch_end_1', torch.tensor([-3, -2, -3]).type_as(acc))

        return result

    def validation_step_dict_return(self, batch, batch_idx):
        acc = self.step(batch, batch_idx)

        logs = {'log_acc1': torch.tensor(12).type_as(acc), 'log_acc2': torch.tensor(7).type_as(acc)}
        pbar = {'pbar_acc1': torch.tensor(17).type_as(acc), 'pbar_acc2': torch.tensor(19).type_as(acc)}
        return {'val_loss': acc, 'log': logs, 'progress_bar': pbar}

    def validation_step_end_basic(self, outputs):
        if self.use_dp or self.use_ddp2:
            pass
        else:
            # only saw 3 batches
            assert len(outputs) == 3
            for batch_out in outputs:
                assert len(batch_out.keys()) == 2
                keys = ['to_batch_end_1', 'to_batch_end_2', 'minimize']
                for key in keys:
                    assert key in batch_out

        result = TrainResult()
        result.pass_to_epoch_end('from_train_step_end', torch.tensor(19))

    def validation_epoch_end_basic(self, outputs):
        if self.use_dp or self.use_ddp2:
            pass
        else:
            # only saw 3 batches
            assert len(outputs) == 3
            for batch_out in outputs:
                assert len(batch_out.keys()) == 2
                keys = ['to_batch_end_1', 'to_batch_end_2']
                for key in keys:
                    assert key in batch_out

    def train_dataloader(self):
        return DataLoader(DummyDataset(), batch_size=3, shuffle=False)

    def val_dataloader(self):
        return DataLoader(DummyDataset(), batch_size=3, shuffle=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0)

    def backward(self, trainer, loss, optimizer, optimizer_idx):
        assert loss == 171.0
        loss.backward()


class DummyDataset(Dataset):

    def __len__(self):
        return 12

    def __getitem__(self, idx):
        return np.array([0.5, 1.0, 2.0])