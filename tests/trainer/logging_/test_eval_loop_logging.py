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
Test logging in the evaluation loop
"""
import collections
import itertools
from unittest import mock
from unittest.mock import call

import numpy as np
import pytest
import torch

from pytorch_lightning import callbacks, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from tests.helpers import BoringModel, RandomDataset


def test__validation_step__log(tmpdir):
    """
    Tests that validation_step can log
    """

    class TestModel(BoringModel):

        def training_step(self, batch, batch_idx):
            out = super().training_step(batch, batch_idx)
            self.log('a', out['loss'], on_step=True, on_epoch=True)
            self.log('a2', 2)
            return out

        def validation_step(self, batch, batch_idx):
            out = super().validation_step(batch, batch_idx)
            self.log('b', out['x'], on_step=True, on_epoch=True)
            return out

    model = TestModel()
    model.validation_step_end = None
    model.validation_epoch_end = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=2,
        log_every_n_steps=1,
        weights_summary=None,
    )
    trainer.fit(model)

    # make sure all the metrics are available for callbacks
    expected_logged_metrics = {
        'a2',
        'a_step',
        'a_epoch',
        'b_step',
        'b_epoch',
        'epoch',
    }
    logged_metrics = set(trainer.logged_metrics.keys())
    assert expected_logged_metrics == logged_metrics

    # we don't want to enable val metrics during steps because it is not something that users should do
    # on purpose DO NOT allow b_step... it's silly to monitor val step metrics
    callback_metrics = set(trainer.callback_metrics.keys())
    expected_cb_metrics = {'a', 'a2', 'b', 'a_epoch', 'b_epoch', 'a_step'}
    assert expected_cb_metrics == callback_metrics


def test__validation_step__epoch_end__log(tmpdir):
    """
    Tests that validation_epoch_end can log
    """

    class TestModel(BoringModel):

        def training_step(self, batch, batch_idx):
            out = super().training_step(batch, batch_idx)
            self.log('a', out['loss'])
            self.log('b', out['loss'], on_step=True, on_epoch=True)
            return out

        def validation_step(self, batch, batch_idx):
            out = super().validation_step(batch, batch_idx)
            self.log('c', out['x'])
            self.log('d', out['x'], on_step=True, on_epoch=True)
            return out

        def validation_epoch_end(self, outputs):
            self.log('g', torch.tensor(2, device=self.device), on_epoch=True)

    model = TestModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=2,
        log_every_n_steps=1,
        weights_summary=None,
    )
    trainer.fit(model)

    # make sure all the metrics are available for loggers
    assert set(trainer.logged_metrics) == {
        'epoch',
        'a',
        'b_step',
        'b_epoch',
        'c',
        'd_step',
        'd_epoch',
        'g',
    }

    assert not trainer.progress_bar_metrics

    # we don't want to enable val metrics during steps because it is not something that users should do
    assert set(trainer.callback_metrics) == {'a', 'b', 'b_epoch', 'c', 'd', 'd_epoch', 'g', 'b_step'}


@pytest.mark.parametrize(['batches', 'log_interval', 'max_epochs'], [(1, 1, 1), (64, 32, 2)])
def test_eval_epoch_logging(tmpdir, batches, log_interval, max_epochs):

    class TestModel(BoringModel):

        def validation_epoch_end(self, outputs):
            self.log('c', torch.tensor(2), on_epoch=True, prog_bar=True, logger=True)
            self.log('d/e/f', 2)

    model = TestModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=batches,
        limit_val_batches=batches,
        max_epochs=max_epochs,
        log_every_n_steps=log_interval,
        weights_summary=None,
    )
    trainer.fit(model)

    # assert the loggers received the expected number
    logged_metrics = set(trainer.logged_metrics)
    assert logged_metrics == {
        'c',
        'd/e/f',
        'epoch',
    }

    pbar_metrics = set(trainer.progress_bar_metrics)
    assert pbar_metrics == {'c'}

    # make sure all the metrics are available for callbacks
    callback_metrics = set(trainer.callback_metrics)
    assert callback_metrics == (logged_metrics | pbar_metrics) - {'epoch'}


def test_eval_float_logging(tmpdir):

    class TestModel(BoringModel):

        def validation_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            self.log('a', 12.0)
            return {"x": loss}

    model = TestModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
        weights_summary=None,
    )
    trainer.fit(model)

    assert set(trainer.logged_metrics) == {'a', 'epoch'}


def test_eval_logging_auto_reduce(tmpdir):

    class TestModel(BoringModel):
        val_losses = []
        manual_epoch_end_mean = None

        def validation_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            self.val_losses.append(loss)
            self.log('val_loss', loss, on_epoch=True, on_step=True, prog_bar=True)
            return {"x": loss}

        def validation_epoch_end(self, outputs) -> None:
            for passed_in, manually_tracked in zip(outputs, self.val_losses):
                assert passed_in['x'] == manually_tracked
            self.manual_epoch_end_mean = torch.stack([x['x'] for x in outputs]).mean()

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=3,
        limit_val_batches=3,
        max_epochs=1,
        log_every_n_steps=1,
        weights_summary=None,
        num_sanity_val_steps=0,
    )
    trainer.fit(model)

    # make sure all the metrics are available for callbacks
    assert set(trainer.callback_metrics) == {'val_loss', 'val_loss_epoch'}

    # make sure values are correct
    assert trainer.logged_metrics['val_loss_epoch'] == model.manual_epoch_end_mean
    assert trainer.callback_metrics['val_loss'] == trainer.logged_metrics['val_loss_step']


@pytest.mark.parametrize(['batches', 'log_interval', 'max_epochs'], [(1, 1, 1), (64, 32, 2)])
def test_eval_epoch_only_logging(tmpdir, batches, log_interval, max_epochs):
    """
    Tests that test_epoch_end can be used to log, and we return them in the results.
    """

    class TestModel(BoringModel):

        def test_epoch_end(self, outputs):
            self.log('c', torch.tensor(2))
            self.log('d/e/f', 2)

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=max_epochs,
        limit_test_batches=batches,
        log_every_n_steps=log_interval,
        weights_summary=None,
    )
    results = trainer.test(model)

    assert len(results) == 1
    assert results[0] == {'c': torch.tensor(2), 'd/e/f': 2}


@pytest.mark.parametrize('suffix', (False, True))
def test_multi_dataloaders_add_suffix_properly(tmpdir, suffix):

    class TestModel(BoringModel):

        def test_step(self, batch, batch_idx, dataloader_idx=0):
            out = super().test_step(batch, batch_idx)
            self.log("test_loss", out['y'], on_step=True, on_epoch=True)
            return out

        def test_dataloader(self):
            if suffix:
                return [
                    torch.utils.data.DataLoader(RandomDataset(32, 64)),
                    torch.utils.data.DataLoader(RandomDataset(32, 64))
                ]
            return super().test_dataloader()

    model = TestModel()
    model.test_epoch_end = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=0,
        limit_val_batches=0,
        limit_test_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
        weights_summary=None,
    )
    results = trainer.test(model)

    for i, r in enumerate(results):
        expected = {'test_loss', 'test_loss_epoch'}
        if suffix:
            expected = {e + f'/dataloader_idx_{i}' for e in expected}
        assert set(r) == expected


def test_log_works_in_val_callback(tmpdir):
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

        def make_logging(self, pl_module, func_name, func_idx, on_steps=[], on_epochs=[], prob_bars=[]):
            self.funcs_called_count[func_name] += 1
            product = [on_steps, on_epochs, prob_bars]
            for idx, (on_step, on_epoch, prog_bar) in enumerate(list(itertools.product(*product))):
                # run logging
                custom_func_name = f"{func_idx}_{idx}_{func_name}"
                pl_module.log(
                    custom_func_name, self.count * func_idx, on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar
                )
                # catch information for verification
                self.callback_funcs_called[func_name].append([self.count * func_idx])
                self.funcs_attr[custom_func_name] = {
                    "on_step": on_step,
                    "on_epoch": on_epoch,
                    "prog_bar": prog_bar,
                    "forked": on_step and on_epoch,
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

        def on_validation_start(self, trainer, pl_module):
            self.make_logging(
                pl_module,
                'on_validation_start',
                1,
                on_steps=self.choices,
                on_epochs=self.choices,
                prob_bars=self.choices
            )

        def on_epoch_start(self, trainer, pl_module):
            if trainer.validating:
                self.make_logging(
                    pl_module,
                    'on_epoch_start',
                    2,
                    on_steps=self.choices,
                    on_epochs=self.choices,
                    prob_bars=self.choices
                )

        def on_validation_epoch_start(self, trainer, pl_module):
            self.make_logging(
                pl_module,
                'on_validation_epoch_start',
                3,
                on_steps=self.choices,
                on_epochs=self.choices,
                prob_bars=self.choices
            )

        def on_batch_end(self, trainer, pl_module):
            self.make_logging(
                pl_module, 'on_batch_end', 6, on_steps=self.choices, on_epochs=self.choices, prob_bars=self.choices
            )

        def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
            self.make_logging(
                pl_module,
                'on_validation_batch_end',
                7,
                on_steps=self.choices,
                on_epochs=self.choices,
                prob_bars=self.choices
            )
            # used to make sure aggregation works fine.
            # we should obtain func[value * c for c in range(1, max_epochs * limit_validation_batches)])
            # with func = np.mean if on_epoch else func = np.max
            self.count += 1

        def on_epoch_end(self, trainer, pl_module):
            if trainer.validating:
                self.make_logging(
                    pl_module, 'on_epoch_end', 8, on_steps=[False], on_epochs=self.choices, prob_bars=self.choices
                )

        def on_validation_epoch_end(self, trainer, pl_module):
            self.make_logging(
                pl_module,
                'on_validation_epoch_end',
                9,
                on_steps=[False],
                on_epochs=self.choices,
                prob_bars=self.choices
            )

    class TestModel(BoringModel):

        def validation_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            self.log('val_loss', loss)

    max_epochs = 1
    model = TestModel()
    model.validation_epoch_end = None
    test_callback = TestCallback()

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=1,
        limit_val_batches=4,
        limit_test_batches=0,
        val_check_interval=0.,
        num_sanity_val_steps=0,
        max_epochs=max_epochs,
        callbacks=[test_callback],
    )
    trainer.fit(model)

    assert test_callback.funcs_called_count["on_epoch_start"] == 1
    # assert test_callback.funcs_called_count["on_batch_start"] == 1
    assert test_callback.funcs_called_count["on_batch_end"] == 1
    assert test_callback.funcs_called_count["on_validation_start"] == 1
    assert test_callback.funcs_called_count["on_validation_epoch_start"] == 1
    # assert test_callback.funcs_called_count["on_validation_batch_start"] == 4
    assert test_callback.funcs_called_count["on_epoch_end"] == 1
    assert test_callback.funcs_called_count["on_validation_batch_end"] == 4
    assert test_callback.funcs_called_count["on_validation_epoch_end"] == 1

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
    trainer.callback_metrics.pop("val_loss")
    for func_name, output_value in trainer.callback_metrics.items():
        # not sure how to handle this now
        if "epoch_0" in func_name:
            func_name = '/'.join(func_name.split('/')[:-1])
            continue

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


def test_log_works_in_test_callback(tmpdir):
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

        def make_logging(self, pl_module, func_name, func_idx, on_steps=[], on_epochs=[], prob_bars=[]):
            original_func_name = func_name[:]
            self.funcs_called_count[original_func_name] += 1
            product = [on_steps, on_epochs, prob_bars]
            for idx, t in enumerate(list(itertools.product(*product))):
                # run logging
                func_name = original_func_name[:]
                on_step, on_epoch, prog_bar = t
                custom_func_name = f"{func_idx}_{idx}_{func_name}"

                pl_module.log(
                    custom_func_name, self.count * func_idx, on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar
                )

                num_dl_ext = ''
                if pl_module._current_dataloader_idx is not None:
                    dl_idx = pl_module._current_dataloader_idx
                    num_dl_ext = f"/dataloader_idx_{dl_idx}"
                    func_name += num_dl_ext

                # catch information for verification
                self.callback_funcs_called[func_name].append([self.count * func_idx])
                self.funcs_attr[custom_func_name + num_dl_ext] = {
                    "on_step": on_step,
                    "on_epoch": on_epoch,
                    "prog_bar": prog_bar,
                    "forked": on_step and on_epoch,
                    "func_name": func_name
                }
                if on_step and on_epoch:
                    self.funcs_attr[f"{custom_func_name}_step" + num_dl_ext] = {
                        "on_step": True,
                        "on_epoch": False,
                        "prog_bar": prog_bar,
                        "forked": False,
                        "func_name": func_name
                    }

                    self.funcs_attr[f"{custom_func_name}_epoch" + num_dl_ext] = {
                        "on_step": False,
                        "on_epoch": True,
                        "prog_bar": prog_bar,
                        "forked": False,
                        "func_name": func_name
                    }

        def on_test_start(self, trainer, pl_module):
            self.make_logging(
                pl_module, 'on_test_start', 1, on_steps=self.choices, on_epochs=self.choices, prob_bars=self.choices
            )

        def on_test_epoch_start(self, trainer, pl_module):
            self.make_logging(
                pl_module,
                'on_test_epoch_start',
                3,
                on_steps=self.choices,
                on_epochs=self.choices,
                prob_bars=self.choices
            )

        def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
            self.make_logging(
                pl_module,
                'on_test_batch_end',
                5,
                on_steps=self.choices,
                on_epochs=self.choices,
                prob_bars=self.choices
            )

            # used to make sure aggregation works fine.
            # we should obtain func[value * c for c in range(1, max_epochs * limit_test_batches)])
            # with func = np.mean if on_epoch else func = np.max
            self.count += 1

        def on_test_epoch_end(self, trainer, pl_module):
            self.make_logging(
                pl_module, 'on_test_epoch_end', 7, on_steps=[False], on_epochs=self.choices, prob_bars=self.choices
            )

    max_epochs = 2
    num_dataloaders = 2

    class TestModel(BoringModel):

        manual_mean = collections.defaultdict(list)

        def test_step(self, batch, batch_idx, dataloader_idx=None):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            self.log('test_loss', loss)
            self.manual_mean[str(dataloader_idx)].append(loss)

        def test_dataloader(self):
            return [torch.utils.data.DataLoader(RandomDataset(32, 64)) for _ in range(num_dataloaders)]

    model = TestModel()
    model.test_epoch_end = None
    test_callback = TestCallback()

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=0,
        limit_test_batches=2,
        val_check_interval=0.,
        num_sanity_val_steps=0,
        max_epochs=max_epochs,
        callbacks=[test_callback],
    )
    trainer.test(model)

    assert test_callback.funcs_called_count["on_test_start"] == 1
    assert test_callback.funcs_called_count["on_test_epoch_start"] == 1
    assert test_callback.funcs_called_count["on_test_batch_end"] == 4
    assert test_callback.funcs_called_count["on_test_epoch_end"] == 1

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
        # Apply mean on values
        if func_attr["on_epoch"] and not func_attr["on_step"]:
            expected_output = np.mean(original_values)
        else:
            expected_output = np.max(original_values)
        return expected_output

    # Make sure the func_name output equals the average from all logged values when on_epoch true
    for dl_idx in range(num_dataloaders):
        key = f"test_loss/dataloader_idx_{dl_idx}"
        assert key in trainer.callback_metrics
        assert torch.stack(model.manual_mean[str(dl_idx)]).mean() == trainer.callback_metrics[key]
        trainer.callback_metrics.pop(key)

    for func_name, output_value in trainer.callback_metrics.items():
        # not sure how to handle this now
        if "epoch_1" in func_name:
            func_name = '/'.join(func_name.split('/')[:-1])
            continue

        if torch.is_tensor(output_value):
            output_value = output_value.item()

        # get func attr
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


@mock.patch("pytorch_lightning.loggers.TensorBoardLogger.log_metrics")
def test_validation_step_log_with_tensorboard(mock_log_metrics, tmpdir):
    """
    This tests make sure we properly log_metrics to loggers
    """

    class ExtendedModel(BoringModel):

        val_losses = []

        def training_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            self.log('train_loss', loss)
            return {"loss": loss}

        def validation_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            self.val_losses.append(loss)
            self.log('valid_loss_0', loss, on_step=True, on_epoch=True)
            self.log('valid_loss_1', loss, on_step=False, on_epoch=True)
            self.log('valid_loss_2', loss, on_step=True, on_epoch=False)
            self.log('valid_loss_3', loss, on_step=False, on_epoch=False)
            return {"val_loss": loss}  # not added to callback_metrics

        def test_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            self.log('test_loss', loss)
            return {"y": loss}

    model = ExtendedModel()
    model.validation_epoch_end = None

    # Initialize a trainer
    trainer = Trainer(
        default_root_dir=tmpdir,
        logger=TensorBoardLogger(tmpdir),
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        max_epochs=2,
        progress_bar_refresh_rate=1,
    )

    # Train the model ⚡
    trainer.fit(model)

    # hp_metric + 2 steps + epoch + 2 steps + epoch
    expected_num_calls = 1 + 2 + 1 + 2 + 1

    assert len(mock_log_metrics.mock_calls) == expected_num_calls
    assert mock_log_metrics.mock_calls[0] == call({'hp_metric': -1}, 0)

    def get_metrics_at_idx(idx):
        mock_calls = list(mock_log_metrics.mock_calls)
        if isinstance(mock_calls[idx].kwargs, dict):
            return mock_calls[idx].kwargs["metrics"]
        else:
            return mock_calls[idx][2]["metrics"]

    expected = {'valid_loss_0_step', 'valid_loss_2'}
    assert set(get_metrics_at_idx(1)) == expected
    assert set(get_metrics_at_idx(2)) == expected

    assert get_metrics_at_idx(1)["valid_loss_0_step"] == model.val_losses[2]
    assert get_metrics_at_idx(2)["valid_loss_0_step"] == model.val_losses[3]

    assert set(get_metrics_at_idx(3)) == {'valid_loss_0_epoch', 'valid_loss_1', 'epoch'}

    assert get_metrics_at_idx(3)["valid_loss_1"] == torch.stack(model.val_losses[2:4]).mean()

    expected = {'valid_loss_0_step', 'valid_loss_2'}
    assert set(get_metrics_at_idx(4)) == expected
    assert set(get_metrics_at_idx(5)) == expected

    assert get_metrics_at_idx(4)["valid_loss_0_step"] == model.val_losses[4]
    assert get_metrics_at_idx(5)["valid_loss_0_step"] == model.val_losses[5]

    assert set(get_metrics_at_idx(6)) == {'valid_loss_0_epoch', 'valid_loss_1', 'epoch'}

    assert get_metrics_at_idx(6)["valid_loss_1"] == torch.stack(model.val_losses[4:]).mean()

    results = trainer.test(model)
    assert set(trainer.callback_metrics) == {
        'train_loss',
        'valid_loss_0_epoch',
        'valid_loss_0',
        'valid_loss_1',
        'test_loss',
    }
    assert set(results[0]) == {'test_loss'}
