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
"""
Test logging in the training loop
"""

import collections
import itertools

import numpy as np
import pytest
import torch

import pytorch_lightning as pl
from pytorch_lightning import callbacks, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from tests.helpers.boring_model import BoringModel, RandomDictDataset
from tests.helpers.runif import RunIf


def test__training_step__log(tmpdir):
    """
    Tests that only training_step can be used
    """

    class TestModel(BoringModel):

        def training_step(self, batch, batch_idx):
            out = super().training_step(batch, batch_idx)
            loss = out['loss']

            # -----------
            # default
            # -----------
            self.log('default', loss)

            # -----------
            # logger
            # -----------
            # on_step T on_epoch F
            self.log('l_s', loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)

            # on_step F on_epoch T
            self.log('l_e', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)

            # on_step T on_epoch T
            self.log('l_se', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

            # -----------
            # pbar
            # -----------
            # on_step T on_epoch F
            self.log('p_s', loss, on_step=True, on_epoch=False, prog_bar=True, logger=False)

            # on_step F on_epoch T
            self.log('p_e', loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)

            # on_step T on_epoch T
            self.log('p_se', loss, on_step=True, on_epoch=True, prog_bar=True, logger=False)

            return loss

    model = TestModel()
    model.val_dataloader = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=2,
        log_every_n_steps=1,
        weights_summary=None,
        callbacks=[ModelCheckpoint(monitor='l_se')],
    )
    trainer.fit(model)

    logged_metrics = set(trainer.logged_metrics)
    assert logged_metrics == {
        'epoch',
        'default',
        'l_e',
        'l_s',
        'l_se_step',
        'l_se_epoch',
    }

    pbar_metrics = set(trainer.progress_bar_metrics)
    assert pbar_metrics == {
        'p_e',
        'p_s',
        'p_se_step',
        'p_se_epoch',
    }

    assert set(trainer.callback_metrics) == (logged_metrics | pbar_metrics | {'p_se', 'l_se'}) - {'epoch'}


def test__training_step__epoch_end__log(tmpdir):
    """
    Tests that training_epoch_end can log
    """

    class TestModel(BoringModel):

        def training_step(self, batch, batch_idx):
            out = super().training_step(batch, batch_idx)
            loss = out['loss']
            self.log('a', loss, on_step=True, on_epoch=True)
            self.log_dict({'a1': loss, 'a2': loss})
            return out

        def training_epoch_end(self, outputs):
            self.log('b1', outputs[0]['loss'])
            self.log('b', outputs[0]['loss'], on_epoch=True, prog_bar=True, logger=True)

    model = TestModel()
    model.val_dataloader = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=2,
        log_every_n_steps=1,
        weights_summary=None,
    )
    trainer.fit(model)

    logged_metrics = set(trainer.logged_metrics)
    assert logged_metrics == {'epoch', 'a_step', 'a_epoch', 'b', 'b1', 'a1', 'a2'}

    pbar_metrics = set(trainer.progress_bar_metrics)
    assert pbar_metrics == {'b'}

    assert set(trainer.callback_metrics) == (logged_metrics | pbar_metrics | {'a'}) - {'epoch'}


@pytest.mark.parametrize(['batches', 'log_interval', 'max_epochs'], [(1, 1, 1), (64, 32, 2)])
def test__training_step__step_end__epoch_end__log(tmpdir, batches, log_interval, max_epochs):
    """
    Tests that training_step_end and training_epoch_end can log
    """

    class TestModel(BoringModel):

        def training_step(self, batch, batch_idx):
            loss = self.step(batch[0])
            self.log('a', loss, on_step=True, on_epoch=True)
            return loss

        def training_step_end(self, out):
            self.log('b', out, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return out

        def training_epoch_end(self, outputs):
            self.log('c', outputs[0]['loss'], on_epoch=True, prog_bar=True, logger=True)
            self.log('d/e/f', 2)

    model = TestModel()
    model.val_dataloader = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=batches,
        limit_val_batches=batches,
        max_epochs=max_epochs,
        log_every_n_steps=log_interval,
        weights_summary=None,
    )
    trainer.fit(model)

    # make sure all the metrics are available for callbacks
    logged_metrics = set(trainer.logged_metrics)
    assert logged_metrics == {'a_step', 'a_epoch', 'b_step', 'b_epoch', 'c', 'd/e/f', 'epoch'}

    pbar_metrics = set(trainer.progress_bar_metrics)
    assert pbar_metrics == {'c', 'b_epoch', 'b_step'}

    assert set(trainer.callback_metrics) == (logged_metrics | pbar_metrics | {'a', 'b'}) - {'epoch'}


@pytest.mark.parametrize(['batches', 'fx', 'result'], [(1, min, 0), (2, max, 1), (11, max, 10)])
def test__training_step__log_max_reduce_fx(tmpdir, batches, fx, result):
    """
    Tests that log works correctly with different tensor types
    """

    class TestModel(BoringModel):

        def training_step(self, batch, batch_idx):
            acc = self.step(batch[0])
            self.log('foo', torch.tensor(batch_idx, dtype=torch.long), on_step=False, on_epoch=True, reduce_fx=fx)
            return acc

        def validation_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            self.log('bar', torch.tensor(batch_idx).float(), on_step=False, on_epoch=True, reduce_fx=fx)
            return {"x": loss}

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=batches,
        limit_val_batches=batches,
        max_epochs=2,
        weights_summary=None,
    )
    trainer.fit(model)

    # make sure types are correct
    assert trainer.logged_metrics['foo'] == result
    assert trainer.logged_metrics['bar'] == result


def test_tbptt_log(tmpdir):
    """
    Tests that only training_step can be used
    """
    truncated_bptt_steps = 2
    sequence_size = 30
    batch_size = 30

    x_seq = torch.rand(batch_size, sequence_size, 1)
    y_seq_list = torch.rand(batch_size, sequence_size, 1).tolist()

    class MockSeq2SeqDataset(torch.utils.data.Dataset):

        def __getitem__(self, i):
            return x_seq, y_seq_list

        def __len__(self):
            return 1

    class TestModel(BoringModel):

        def __init__(self):
            super().__init__()
            self.test_hidden = None
            self.layer = torch.nn.Linear(2, 2)

        def training_step(self, batch, batch_idx, hiddens):
            assert hiddens == self.test_hidden, "Hidden state not persistent between tbptt steps"
            self.test_hidden = torch.rand(1)

            x_tensor, y_list = batch
            assert x_tensor.shape[1] == truncated_bptt_steps, "tbptt split Tensor failed"

            y_tensor = torch.tensor(y_list, dtype=x_tensor.dtype)
            assert y_tensor.shape[1] == truncated_bptt_steps, "tbptt split list failed"

            pred = self(x_tensor.view(batch_size, truncated_bptt_steps))
            loss = torch.nn.functional.mse_loss(pred, y_tensor.view(batch_size, truncated_bptt_steps))

            self.log('a', loss, on_epoch=True)

            return {'loss': loss, 'hiddens': self.test_hidden}

        def on_train_epoch_start(self) -> None:
            self.test_hidden = None

        def train_dataloader(self):
            return torch.utils.data.DataLoader(
                dataset=MockSeq2SeqDataset(),
                batch_size=batch_size,
                shuffle=False,
                sampler=None,
            )

    model = TestModel()
    model.training_epoch_end = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=10,
        limit_val_batches=0,
        truncated_bptt_steps=truncated_bptt_steps,
        max_epochs=2,
        log_every_n_steps=2,
        weights_summary=None,
    )
    trainer.fit(model)

    assert set(trainer.logged_metrics) == {'a_step', 'a_epoch', 'epoch'}


def test_different_batch_types_for_sizing(tmpdir):

    class TestModel(BoringModel):

        def training_step(self, batch, batch_idx):
            assert isinstance(batch, dict)
            a = batch['a']
            acc = self.step(a)
            self.log('a', {'d1': 2, 'd2': torch.tensor(1)}, on_step=True, on_epoch=True)
            return acc

        def validation_step(self, batch, batch_idx):
            assert isinstance(batch, dict)
            a = batch['a']
            output = self.layer(a)
            loss = self.loss(batch, output)
            self.log('n', {'d3': 2, 'd4': torch.tensor(1)}, on_step=True, on_epoch=True)
            return {"x": loss}

        def train_dataloader(self):
            return torch.utils.data.DataLoader(RandomDictDataset(32, 64), batch_size=32)

        def val_dataloader(self):
            return torch.utils.data.DataLoader(RandomDictDataset(32, 64), batch_size=32)

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=1,
        limit_val_batches=2,
        max_epochs=1,
        weights_summary=None,
    )
    trainer.fit(model)

    assert set(trainer.logged_metrics) == {'a_step', 'a_epoch', 'n_step', 'n_epoch', 'epoch'}


def test_log_works_in_train_callback(tmpdir):
    """
    Tests that log can be called within callback
    """

    class TestCallback(callbacks.Callback):

        # helpers
        count = 1
        choices = [False, True]
        # used to compute expected values
        callback_funcs_called = collections.defaultdict(list)
        funcs_called_count = collections.defaultdict(int)
        funcs_attr = {}

        def make_logging(
            self, pl_module: pl.LightningModule, func_name, func_idx, on_steps=[], on_epochs=[], prob_bars=[]
        ):
            self.funcs_called_count[func_name] += 1
            iterate = list(itertools.product(*[on_steps, on_epochs, prob_bars]))
            for idx, (on_step, on_epoch, prog_bar) in enumerate(iterate):
                # run logging
                custom_func_name = f"{func_idx}_{idx}_{func_name}"
                pl_module.log(
                    custom_func_name, self.count * func_idx, on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar
                )

                # catch information for verification

                # on on_train_start is outside the main loop. Won't be called
                if func_name == "on_train_start":
                    self.callback_funcs_called[func_name].append([self.count * func_idx])

                # Saved only values from second epoch, so we can compute its mean or latest.
                if pl_module.trainer.current_epoch == 1:
                    self.callback_funcs_called[func_name].append([self.count * func_idx])

                forked = on_step and on_epoch

                self.funcs_attr[custom_func_name] = {
                    "on_step": on_step,
                    "on_epoch": on_epoch,
                    "prog_bar": prog_bar,
                    "forked": forked,
                    "func_name": func_name
                }

                if on_step and on_epoch:
                    self.funcs_attr[f"{custom_func_name}_step"] = {
                        "on_step": True,
                        "on_epoch": False,
                        "prog_bar": prog_bar,
                        "forked": False,
                        "func_name": func_name
                    }

                    self.funcs_attr[f"{custom_func_name}_epoch"] = {
                        "on_step": False,
                        "on_epoch": True,
                        "prog_bar": prog_bar,
                        "forked": False,
                        "func_name": func_name
                    }

        def on_train_start(self, trainer, pl_module):
            self.make_logging(
                pl_module, 'on_train_start', 1, on_steps=self.choices, on_epochs=self.choices, prob_bars=self.choices
            )

        def on_epoch_start(self, trainer, pl_module):
            self.make_logging(
                pl_module, 'on_epoch_start', 2, on_steps=self.choices, on_epochs=self.choices, prob_bars=self.choices
            )

        def on_train_epoch_start(self, trainer, pl_module):
            self.make_logging(
                pl_module,
                'on_train_epoch_start',
                3,
                on_steps=self.choices,
                on_epochs=self.choices,
                prob_bars=self.choices
            )

        def on_batch_end(self, trainer, pl_module):
            self.make_logging(
                pl_module, 'on_batch_end', 6, on_steps=self.choices, on_epochs=self.choices, prob_bars=self.choices
            )

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
            self.make_logging(
                pl_module,
                'on_train_batch_end',
                7,
                on_steps=self.choices,
                on_epochs=self.choices,
                prob_bars=self.choices
            )
            # used to make sure aggregation works fine.
            # we should obtain func[value * c for c in range(1, max_epochs * limit_train_batches)])
            # with func = np.mean if on_epoch else func = np.max
            self.count += 1

        def on_train_epoch_end(self, trainer, pl_module):
            self.make_logging(
                pl_module, 'on_train_epoch_end', 8, on_steps=[False], on_epochs=self.choices, prob_bars=self.choices
            )

        def on_epoch_end(self, trainer, pl_module):
            self.make_logging(
                pl_module, 'on_epoch_end', 9, on_steps=[False], on_epochs=self.choices, prob_bars=self.choices
            )

    class TestModel(BoringModel):

        manual_loss = []

        def training_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            self.manual_loss.append(loss)
            self.log('train_loss', loss)
            return {"loss": loss}

    max_epochs = 2
    limit_train_batches = 2
    model = TestModel()
    test_callback = TestCallback()

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=limit_train_batches,
        limit_val_batches=0,
        limit_test_batches=0,
        val_check_interval=0.,
        num_sanity_val_steps=0,
        max_epochs=max_epochs,
        callbacks=[test_callback]
    )
    trainer.fit(model)

    assert test_callback.funcs_called_count["on_train_start"] == 1
    assert test_callback.funcs_called_count["on_epoch_start"] == 2
    assert test_callback.funcs_called_count["on_train_epoch_start"] == 2
    assert test_callback.funcs_called_count["on_batch_end"] == 4
    assert test_callback.funcs_called_count["on_epoch_end"] == 2
    assert test_callback.funcs_called_count["on_train_batch_end"] == 4
    assert test_callback.funcs_called_count["on_epoch_end"] == 2
    assert test_callback.funcs_called_count["on_train_epoch_end"] == 2

    # Make sure the func_name exists within callback_metrics. If not, we missed some
    callback_metrics_keys = [*trainer.callback_metrics.keys()]
    for func_name in test_callback.callback_funcs_called.keys():
        is_in = False
        for callback_metrics_key in callback_metrics_keys:
            if func_name in callback_metrics_key:
                is_in = True
        assert is_in, (func_name, callback_metrics_keys)

    # function used to describe expected return logic
    def get_expected_output(func_attr, original_values):
        if func_attr["on_epoch"] and not func_attr["on_step"]:
            # Apply mean on values
            expected_output = np.mean(original_values)
        else:
            # Keep the latest value
            expected_output = np.max(original_values)
        return expected_output

    # Make sure the func_name output equals the average from all logged values when on_epoch true
    # pop extra keys
    assert trainer.logged_metrics["train_loss"] == model.manual_loss[-1]
    assert trainer.callback_metrics["train_loss"] == model.manual_loss[-1]
    trainer.callback_metrics.pop("train_loss")

    for func_name, output_value in trainer.callback_metrics.items():
        if torch.is_tensor(output_value):
            output_value = output_value.item()
        # get creation attr
        func_attr = test_callback.funcs_attr[func_name]

        # retrived orginal logged values
        original_values = test_callback.callback_funcs_called[func_attr["func_name"]]

        # compute expected output and compare to actual one
        expected_output = get_expected_output(func_attr, original_values)
        assert float(output_value) == float(expected_output)

    for func_name, func_attr in test_callback.funcs_attr.items():
        if func_attr["prog_bar"] and (func_attr["on_step"] or func_attr["on_epoch"]) and not func_attr["forked"]:
            assert func_name in trainer.logger_connector.progress_bar_metrics
        else:
            assert func_name not in trainer.logger_connector.progress_bar_metrics


def test_logging_sync_dist_true_cpu(tmpdir):
    """
    Tests to ensure that the sync_dist flag works with CPU (should just return the original value)
    """
    fake_result = 1

    class TestModel(BoringModel):

        def training_step(self, batch, batch_idx):
            acc = self.step(batch[0])
            self.log('foo', torch.tensor(fake_result), on_step=False, on_epoch=True, sync_dist=True, sync_dist_op='sum')
            self.log('foo_2', 2, on_step=False, on_epoch=True, sync_dist=True, sync_dist_op='sum')
            return acc

        def validation_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            self.log('bar', torch.tensor(fake_result), on_step=False, on_epoch=True, sync_dist=True, sync_dist_op='sum')
            return {"x": loss}

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=1,
        limit_val_batches=1,
        max_epochs=2,
        weights_summary=None,
    )
    trainer.fit(model)

    assert trainer.logged_metrics['foo'] == fake_result
    assert trainer.logged_metrics['foo_2'] == 2
    assert trainer.logged_metrics['bar'] == fake_result


@RunIf(min_gpus=2, special=True)
def test_logging_sync_dist_true_ddp(tmpdir):
    """
    Tests to ensure that the sync_dist flag works with ddp
    """

    class TestLoggingSyncDistModel(BoringModel):

        def training_step(self, batch, batch_idx):
            acc = self.step(batch[0])
            self.log('foo', 1, on_step=False, on_epoch=True, sync_dist=True, sync_dist_op='SUM')
            self.log('cho', acc, on_step=False, on_epoch=True)
            return acc

        def validation_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            self.log('bar', 2, on_step=False, on_epoch=True, sync_dist=True, sync_dist_op='AVG')
            return {"x": loss}

    model = TestLoggingSyncDistModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=1,
        limit_val_batches=1,
        max_epochs=2,
        weights_summary=None,
        accelerator="ddp",
        gpus=2,
        profiler="pytorch"
    )
    trainer.fit(model)

    assert trainer.logged_metrics['foo'] == 2
    assert trainer.logged_metrics['bar'] == 2


@RunIf(min_gpus=1)
def test_logging_sync_dist_true_gpu(tmpdir):
    """
    Tests to ensure that the sync_dist flag works with GPU (should just return the original value)
    """
    fake_result = 1

    class TestModel(BoringModel):

        def training_step(self, batch, batch_idx):
            acc = self.step(batch[0])
            self.log('foo', torch.tensor(fake_result), on_step=False, on_epoch=True, sync_dist=True, sync_dist_op='sum')
            return acc

        def validation_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            self.log('bar', torch.tensor(fake_result), on_step=False, on_epoch=True, sync_dist=True, sync_dist_op='sum')
            return {"x": loss}

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=1,
        limit_val_batches=1,
        max_epochs=2,
        gpus=1,
        weights_summary=None,
    )
    trainer.fit(model)

    assert trainer.logged_metrics['foo'] == fake_result
    assert trainer.logged_metrics['bar'] == fake_result


def test_progress_bar_dict_contains_values_on_train_epoch_end(tmpdir):

    class TestModel(BoringModel):

        def training_step(self, *args):
            self.log("foo", torch.tensor(self.current_epoch), on_step=False, on_epoch=True, prog_bar=True)
            return super().training_step(*args)

        def on_train_epoch_end(self, *_):
            self.on_train_epoch_end_called = True
            self.epoch_end_called = True
            self.log(
                'foo_2',
                torch.tensor(self.current_epoch),
                prog_bar=True,
                on_epoch=True,
                sync_dist=True,
                sync_dist_op='sum'
            )

        def on_epoch_end(self):
            self.epoch_end_called = True
            assert self.trainer.progress_bar_dict["foo"] == self.current_epoch
            assert self.trainer.progress_bar_dict["foo_2"] == self.current_epoch

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_train_batches=1,
        limit_val_batches=0,
        checkpoint_callback=False,
        logger=False,
        weights_summary=None,
        progress_bar_refresh_rate=0,
    )
    model = TestModel()
    trainer.fit(model)
    assert model.epoch_end_called
    assert model.on_train_epoch_end_called


def test_logging_in_callbacks_with_log_function(tmpdir):
    """
    Tests ensure self.log can be used directly in callbacks.
    """

    class LoggingCallback(callbacks.Callback):

        def on_train_start(self, trainer, pl_module):
            self.log("on_train_start", 1)

        def on_train_epoch_start(self, trainer, pl_module):
            self.log("on_train_epoch_start", 2)

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
            self.log("on_train_batch_end", 3)

        def on_batch_end(self, trainer, pl_module):
            self.log("on_batch_end", 4)

        def on_epoch_end(self, trainer, pl_module):
            self.log("on_epoch_end", 5)

        def on_train_epoch_end(self, trainer, pl_module, outputs):
            self.log("on_train_epoch_end", 6)
            self.callback_metrics = trainer.logger_connector.callback_metrics

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=1,
        limit_val_batches=1,
        max_epochs=1,
        weights_summary=None,
        callbacks=[LoggingCallback()]
    )
    trainer.fit(model)

    expected = {
        'on_train_start': 1,
        'on_train_epoch_start': 2,
        'on_train_batch_end': 3,
        'on_batch_end': 4,
        'on_epoch_end': 5,
        'on_train_epoch_end': 6
    }
    assert trainer.callback_metrics == expected


@RunIf(min_gpus=1)
def test_metric_are_properly_reduced(tmpdir):

    class TestingModel(BoringModel):

        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            self.val_acc = pl.metrics.Accuracy()

        def training_step(self, batch, batch_idx):
            output = super().training_step(batch, batch_idx)
            self.log("train_loss", output["loss"])
            return output

        def validation_step(self, batch, batch_idx):
            preds = torch.tensor([[0.9, 0.1]], device=self.device)
            targets = torch.tensor([1], device=self.device)
            if batch_idx < 8:
                preds = torch.tensor([[0.1, 0.9]], device=self.device)
            self.val_acc(preds, targets)
            self.log('val_acc', self.val_acc, on_step=True, on_epoch=True)
            return super().validation_step(batch, batch_idx)

    early_stop = EarlyStopping(monitor='val_acc', mode='max')

    checkpoint = ModelCheckpoint(
        monitor='val_acc',
        save_last=True,
        save_top_k=2,
        mode='max',
    )

    model = TestingModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        gpus=1,
        max_epochs=2,
        limit_train_batches=5,
        limit_val_batches=32,
        callbacks=[early_stop, checkpoint]
    )
    trainer.fit(model)

    assert trainer.callback_metrics["val_acc"] == 8 / 32.
    assert "train_loss" in trainer.callback_metrics


@pytest.mark.parametrize('value', [None, {'a': {'b': None}}])
def test_log_none_raises(tmpdir, value):

    class TestModel(BoringModel):

        def training_step(self, *args):
            self.log("foo", value)

    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=1)
    model = TestModel()
    with pytest.raises(ValueError, match=rf"self.log\(foo, {value}\)` was called"):
        trainer.fit(model)
