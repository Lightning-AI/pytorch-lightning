import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from pytorch_lightning import TrainResult, EvalResult
from pytorch_lightning.core.lightning import LightningModule


class DeterministicModel(LightningModule):

    def __init__(self, weights=None):
        super().__init__()

        self.training_step_called = False
        self.training_step_end_called = False
        self.training_epoch_end_called = False

        self.validation_step_called = False
        self.validation_step_end_called = False
        self.validation_epoch_end_called = False

        self.assert_backward = True

        self.l1 = nn.Linear(2, 3, bias=False)
        if weights is None:
            weights = torch.tensor([
                [4, 3, 5],
                [10, 11, 13]
            ]).float()
            p = torch.nn.Parameter(weights, requires_grad=True)
            self.l1.weight = p

    def forward(self, x):
        return self.l1(x)

    def step(self, batch, batch_idx):
        x = batch
        bs = x.size(0)
        y_hat = self.l1(x)

        test_hat = y_hat.cpu().detach()
        assert torch.all(test_hat[:, 0] == 15.0)
        assert torch.all(test_hat[:, 1] == 42.0)
        out = y_hat.sum()
        assert out == (42.0 * bs) + (15.0 * bs)

        return out

    def assert_graph_count(self, result, count=1):
        counts = self.count_num_graphs(result)
        assert counts == count

    def count_num_graphs(self, result, num_graphs=0):
        for k, v in result.items():
            if isinstance(v, torch.Tensor) and v.grad_fn is not None:
                num_graphs += 1
            if isinstance(v, dict):
                num_graphs += self.count_num_graphs(v)

        return num_graphs

    # ---------------------------
    # scalar return
    # ---------------------------
    def training_step_scalar_return(self, batch, batch_idx):
        acc = self.step(batch, batch_idx)
        self.training_step_called = True
        return acc

    def training_step_end_scalar(self, output):
        self.training_step_end_called = True

        # make sure loss has the grad
        assert isinstance(output, torch.Tensor)
        assert output.grad_fn is not None

        # make sure nothing else has grads
        assert self.count_num_graphs({'loss': output}) == 1

        assert output == 171

        return output

    def training_epoch_end_scalar(self, outputs):
        """
        There should be an array of scalars without graphs that are all 171 (4 of them)
        """
        self.training_epoch_end_called = True

        if self.use_dp or self.use_ddp2:
            pass
        else:
            # only saw 4 batches
            assert len(outputs) == 4
            for batch_out in outputs:
                assert batch_out == 171
                assert batch_out.grad_fn is None
                assert isinstance(batch_out, torch.Tensor)

        prototype_loss = outputs[0]
        return prototype_loss

    def training_step_no_default_callbacks_for_train_loop(self, batch, batch_idx):
        """
        Early stop and checkpoint only on these values
        """
        acc = self.step(batch, batch_idx)
        result = TrainResult(minimize=acc)
        assert 'early_step_on' not in result
        assert 'checkpoint_on' in result
        return result

    def training_step_no_callbacks_result_obj(self, batch, batch_idx):
        """
        Early stop and checkpoint only on these values
        """
        acc = self.step(batch, batch_idx)
        result = TrainResult(minimize=acc, checkpoint_on=False)
        assert 'early_step_on' not in result
        assert 'checkpoint_on' not in result
        return result

    def training_step_result_log_epoch_and_step_for_callbacks(self, batch, batch_idx):
        """
        Early stop and checkpoint only on these values
        """
        acc = self.step(batch, batch_idx)

        self.assert_backward = False
        losses = [20, 19, 18, 10, 15, 14, 9, 11, 11, 20]
        idx = self.current_epoch
        loss = acc + losses[idx]
        result = TrainResult(minimize=loss, early_stop_on=loss, checkpoint_on=loss)
        return result

    def training_step_result_log_step_only(self, batch, batch_idx):
        acc = self.step(batch, batch_idx)
        result = TrainResult(minimize=acc)

        # step only metrics
        result.log(f'step_log_and_pbar_acc1_b{batch_idx}', torch.tensor(11).type_as(acc), prog_bar=True)
        result.log(f'step_log_acc2_b{batch_idx}', torch.tensor(12).type_as(acc))
        result.log(f'step_pbar_acc3_b{batch_idx}', torch.tensor(13).type_as(acc), logger=False, prog_bar=True)

        self.training_step_called = True
        return result

    def training_step_result_log_epoch_only(self, batch, batch_idx):
        acc = self.step(batch, batch_idx)
        result = TrainResult(minimize=acc)

        result.log(f'epoch_log_and_pbar_acc1_e{self.current_epoch}', torch.tensor(14).type_as(acc),
                   on_epoch=True, prog_bar=True, on_step=False)
        result.log(f'epoch_log_acc2_e{self.current_epoch}', torch.tensor(15).type_as(acc),
                   on_epoch=True, on_step=False)
        result.log(f'epoch_pbar_acc3_e{self.current_epoch}', torch.tensor(16).type_as(acc),
                   on_epoch=True, logger=False, prog_bar=True, on_step=False)

        self.training_step_called = True
        return result

    def training_step_result_log_epoch_and_step(self, batch, batch_idx):
        acc = self.step(batch, batch_idx)
        result = TrainResult(minimize=acc)

        val_1 = (5 + batch_idx) * (self.current_epoch + 1)
        val_2 = (6 + batch_idx) * (self.current_epoch + 1)
        val_3 = (7 + batch_idx) * (self.current_epoch + 1)
        result.log('step_epoch_log_and_pbar_acc1', torch.tensor(val_1).type_as(acc),
                   on_epoch=True, prog_bar=True)
        result.log('step_epoch_log_acc2', torch.tensor(val_2).type_as(acc),
                   on_epoch=True)
        result.log('step_epoch_pbar_acc3', torch.tensor(val_3).type_as(acc),
                   on_epoch=True, logger=False, prog_bar=True)

        self.training_step_called = True
        return result

    def training_epoch_end_return_for_log_epoch_and_step(self, result):
        """
        There should be an array of scalars without graphs that are all 171 (4 of them)
        """
        self.training_epoch_end_called = True

        if self.use_dp or self.use_ddp2:
            pass
        else:
            # only saw 4 batches
            assert isinstance(result, TrainResult)

        result.step_step_epoch_log_and_pbar_acc1 = result.step_step_epoch_log_and_pbar_acc1.prod()
        result.epoch_step_epoch_log_and_pbar_acc1 = result.epoch_step_epoch_log_and_pbar_acc1.prod()
        result.step_step_epoch_log_acc2 = result.step_step_epoch_log_acc2.prod()
        result.epoch_step_epoch_log_acc2 = result.epoch_step_epoch_log_acc2.prod()
        result.step_step_epoch_pbar_acc3 = result.step_step_epoch_pbar_acc3.prod()
        result.epoch_step_epoch_pbar_acc3 = result.epoch_step_epoch_pbar_acc3.prod()
        result.log('epoch_end_log_acc', torch.tensor(1212).type_as(result.epoch_step_epoch_log_acc2),
                   logger=True, on_epoch=True)
        result.log('epoch_end_pbar_acc', torch.tensor(1213).type_as(result.epoch_step_epoch_log_acc2),
                   logger=False, prog_bar=True, on_epoch=True)
        result.log('epoch_end_log_pbar_acc', torch.tensor(1214).type_as(result.epoch_step_epoch_log_acc2),
                   logger=True, prog_bar=True, on_epoch=True)
        return result

    # --------------------------
    # EvalResults
    # --------------------------
    def validation_step_result_callbacks(self, batch, batch_idx):
        acc = self.step(batch, batch_idx)

        self.assert_backward = False
        losses = [20, 19, 20, 21, 22, 23]
        idx = self.current_epoch
        loss = acc + losses[idx]
        result = EvalResult(early_stop_on=loss, checkpoint_on=loss)

        self.validation_step_called = True
        return result

    def validation_step_result_no_callbacks(self, batch, batch_idx):
        acc = self.step(batch, batch_idx)

        self.assert_backward = False
        losses = [20, 19, 20, 21, 22, 23, 50, 50, 50, 50, 50, 50]
        idx = self.current_epoch
        loss = acc + losses[idx]

        result = EvalResult(checkpoint_on=loss)

        self.validation_step_called = True
        return result

    def validation_step_result_only_epoch_metrics(self, batch, batch_idx):
        """
        Only track epoch level metrics
        """
        acc = self.step(batch, batch_idx)
        result = EvalResult(checkpoint_on=acc, early_stop_on=acc)

        # step only metrics
        result.log('no_val_no_pbar', torch.tensor(11 + batch_idx).type_as(acc), prog_bar=False, logger=False)
        result.log('val_step_log_acc', torch.tensor(11 + batch_idx).type_as(acc), prog_bar=False, logger=True)
        result.log('val_step_log_pbar_acc', torch.tensor(12 + batch_idx).type_as(acc), prog_bar=True, logger=True)
        result.log('val_step_pbar_acc', torch.tensor(13 + batch_idx).type_as(acc), prog_bar=True, logger=False)

        self.validation_step_called = True
        return result

    def validation_step_result_only_step_metrics(self, batch, batch_idx):
        """
        Only track epoch level metrics
        """
        acc = self.step(batch, batch_idx)
        result = EvalResult(checkpoint_on=acc, early_stop_on=acc)

        # step only metrics
        result.log('no_val_no_pbar', torch.tensor(11 + batch_idx).type_as(acc),
                   prog_bar=False, logger=False, on_epoch=False, on_step=True)
        result.log('val_step_log_acc', torch.tensor(11 + batch_idx).type_as(acc),
                   prog_bar=False, logger=True, on_epoch=False, on_step=True)
        result.log('val_step_log_pbar_acc', torch.tensor(12 + batch_idx).type_as(acc),
                   prog_bar=True, logger=True, on_epoch=False, on_step=True)
        result.log('val_step_pbar_acc', torch.tensor(13 + batch_idx).type_as(acc),
                   prog_bar=True, logger=False, on_epoch=False, on_step=True)
        result.log('val_step_batch_idx', torch.tensor(batch_idx).type_as(acc),
                   prog_bar=True, logger=True, on_epoch=False, on_step=True)

        self.validation_step_called = True
        return result

    def validation_step_result_epoch_step_metrics(self, batch, batch_idx):
        """
        Only track epoch level metrics
        """
        acc = self.step(batch, batch_idx)
        result = EvalResult(checkpoint_on=acc, early_stop_on=acc)

        # step only metrics
        result.log('no_val_no_pbar', torch.tensor(11 + batch_idx).type_as(acc),
                   prog_bar=False, logger=False, on_epoch=True, on_step=True)
        result.log('val_step_log_acc', torch.tensor(11 + batch_idx).type_as(acc),
                   prog_bar=False, logger=True, on_epoch=True, on_step=True)
        result.log('val_step_log_pbar_acc', torch.tensor(12 + batch_idx).type_as(acc),
                   prog_bar=True, logger=True, on_epoch=True, on_step=True)
        result.log('val_step_pbar_acc', torch.tensor(13 + batch_idx).type_as(acc),
                   prog_bar=True, logger=False, on_epoch=True, on_step=True)
        result.log('val_step_batch_idx', torch.tensor(batch_idx).type_as(acc),
                   prog_bar=True, logger=True, on_epoch=True, on_step=True)

        self.validation_step_called = True
        return result

    def validation_step_for_epoch_end_result(self, batch, batch_idx):
        """
        EvalResult flows to epoch end (without step_end)
        """
        acc = self.step(batch, batch_idx)
        result = EvalResult(checkpoint_on=acc, early_stop_on=acc)

        # step only metrics
        result.log('val_step_metric', torch.tensor(batch_idx).type_as(acc),
                   prog_bar=True, logger=True, on_epoch=True, on_step=False)
        result.log('batch_idx', torch.tensor(batch_idx).type_as(acc),
                   prog_bar=True, logger=True, on_epoch=True, on_step=False)

        self.validation_step_called = True
        return result

    def validation_epoch_end_result(self, result):
        self.validation_epoch_end_called = True

        if self.trainer.running_sanity_check:
            assert len(result.batch_idx) == 2
        else:
            assert len(result.batch_idx) == self.trainer.limit_val_batches

        expected_val = result.val_step_metric.sum() / len(result.batch_idx)
        result.val_step_metric = result.val_step_metric.mean()
        result.batch_idx = result.batch_idx.mean()
        assert result.val_step_metric == expected_val

        result.log('val_epoch_end_metric', torch.tensor(189).type_as(result.val_step_metric), prog_bar=True)

        return result

    # --------------------------
    # dictionary returns
    # --------------------------
    def training_step_dict_return(self, batch, batch_idx):
        acc = self.step(batch, batch_idx)

        logs = {'log_acc1': torch.tensor(12).type_as(acc), 'log_acc2': torch.tensor(7).type_as(acc)}
        pbar = {'pbar_acc1': torch.tensor(17).type_as(acc), 'pbar_acc2': torch.tensor(19).type_as(acc)}

        self.training_step_called = True
        return {'loss': acc, 'log': logs, 'progress_bar': pbar, 'train_step_test': torch.tensor(549).type_as(acc)}

    def training_step_for_step_end_dict(self, batch, batch_idx):
        """sends outputs to training_batch_end"""
        acc = self.step(batch, batch_idx)

        logs = {'log_acc1': torch.tensor(12).type_as(acc), 'log_acc2': torch.tensor(7).type_as(acc)}
        pbar = {'pbar_acc1': torch.tensor(17).type_as(acc), 'pbar_acc2': torch.tensor(19).type_as(acc)}

        self.training_step_called = True
        result = {'loss': acc}
        result.update(logs)
        result.update(pbar)
        return result

    def training_step_end_dict(self, output):
        self.training_step_end_called = True

        # make sure loss has the grad
        assert 'loss' in output
        assert output['loss'].grad_fn is not None

        # make sure nothing else has grads
        assert self.count_num_graphs(output) == 1

        # make sure the other keys are there
        assert 'log_acc1' in output
        assert 'log_acc2' in output
        assert 'pbar_acc1' in output
        assert 'pbar_acc2' in output

        logs = {'log_acc1': output['log_acc1'] + 2, 'log_acc2': output['log_acc2'] + 2}
        pbar = {'pbar_acc1': output['pbar_acc1'] + 2, 'pbar_acc2': output['pbar_acc2'] + 2}

        acc = output['loss']
        return {'loss': acc, 'log': logs, 'progress_bar': pbar, 'train_step_end': acc}

    def training_epoch_end_dict(self, outputs):
        self.training_epoch_end_called = True

        if self.use_dp or self.use_ddp2:
            pass
        else:
            # only saw 4 batches
            assert len(outputs) == 4
            for batch_out in outputs:
                assert len(batch_out.keys()) == 4
                assert self.count_num_graphs(batch_out) == 0
                last_key = 'train_step_end' if self.training_step_end_called else 'train_step_test'
                keys = ['loss', 'log', 'progress_bar', last_key]
                for key in keys:
                    assert key in batch_out

        prototype_loss = outputs[0]['loss']
        logs = {'epoch_end_log_1': torch.tensor(178).type_as(prototype_loss)}
        pbar = {'epoch_end_pbar_1': torch.tensor(234).type_as(prototype_loss)}

        return {'log': logs, 'progress_bar': pbar}

    def validation_step_no_return(self, batch, batch_idx):
        self.validation_step_called = True
        acc = self.step(batch, batch_idx)

    def validation_step_scalar_return(self, batch, batch_idx):
        self.validation_step_called = True
        acc = self.step(batch, batch_idx)
        return acc

    def validation_step_arbitary_dict_return(self, batch, batch_idx):
        self.validation_step_called = True
        acc = self.step(batch, batch_idx)
        return {'some': acc, 'value': 'a'}

    def validation_step_dict_return(self, batch, batch_idx):
        self.validation_step_called = True
        acc = self.step(batch, batch_idx)

        logs = {'log_acc1': torch.tensor(12 + batch_idx).type_as(acc), 'log_acc2': torch.tensor(7).type_as(acc)}
        pbar = {'pbar_acc1': torch.tensor(17).type_as(acc), 'pbar_acc2': torch.tensor(19).type_as(acc)}
        return {'val_loss': acc, 'log': logs, 'progress_bar': pbar}

    def validation_step_end_no_return(self, val_step_output):
        assert len(val_step_output) == 3
        assert val_step_output['val_loss'] == 171
        assert val_step_output['log']['log_acc1'] >= 12
        assert val_step_output['progress_bar']['pbar_acc1'] == 17
        self.validation_step_end_called = True

    def validation_step_end(self, val_step_output):
        assert len(val_step_output) == 3
        assert val_step_output['val_loss'] == 171
        assert val_step_output['log']['log_acc1'] >= 12
        assert val_step_output['progress_bar']['pbar_acc1'] == 17
        self.validation_step_end_called = True

        val_step_output['val_step_end'] = torch.tensor(1802)

        return val_step_output

    def validation_epoch_end(self, outputs):
        assert len(outputs) == self.trainer.num_val_batches[0]

        for i, out in enumerate(outputs):
            assert out['log']['log_acc1'] >= 12 + i

        self.validation_epoch_end_called = True

        result = outputs[-1]
        result['val_epoch_end'] = torch.tensor(1233)
        return result

    # -----------------------------
    # DATA
    # -----------------------------
    def train_dataloader(self):
        return DataLoader(DummyDataset(), batch_size=3, shuffle=False)

    def val_dataloader(self):
        return DataLoader(DummyDataset(), batch_size=3, shuffle=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0)

    def configure_optimizers__lr_on_plateau_epoch(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        scheduler = {'scheduler': lr_scheduler, 'interval': 'epoch', 'monitor': 'epoch_end_log_1'}
        return [optimizer], [scheduler]

    def configure_optimizers__lr_on_plateau_step(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        scheduler = {'scheduler': lr_scheduler, 'interval': 'step', 'monitor': 'pbar_acc1'}
        return [optimizer], [scheduler]

    def backward(self, trainer, loss, optimizer, optimizer_idx):
        if self.assert_backward:
            if self.trainer.precision == 16:
                assert loss > 171 * 1000
            else:
                assert loss == 171.0

        loss.backward()


class DummyDataset(Dataset):

    def __len__(self):
        return 12

    def __getitem__(self, idx):
        return torch.tensor([0.5, 1.0, 2.0])
