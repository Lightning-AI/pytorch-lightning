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
Tests to ensure that the training loop works with a dict (1.0)
"""
from pytorch_lightning import Trainer
from pytorch_lightning import callbacks, seed_everything
from tests.base.deterministic_model import DeterministicModel
from tests.base import SimpleModule, BoringModel, create_scriptable_callback
import os
import torch
import pytest


def test__validation_step__log(tmpdir):
    """
    Tests that validation_step can log
    """
    os.environ['PL_DEV_DEBUG'] = '1'

    class TestModel(DeterministicModel):
        def training_step(self, batch, batch_idx):
            acc = self.step(batch, batch_idx)
            acc = acc + batch_idx
            self.log('a', acc, on_step=True, on_epoch=True)
            self.log('a2', 2)

            self.training_step_called = True
            return acc

        def validation_step(self, batch, batch_idx):
            acc = self.step(batch, batch_idx)
            acc = acc + batch_idx
            self.log('b', acc, on_step=True, on_epoch=True)
            self.training_step_called = True

        def backward(self, loss, optimizer, optimizer_idx):
            loss.backward()

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
        'b_step/epoch_0',
        'b_step/epoch_1',
        'b_epoch',
        'epoch',
    }
    logged_metrics = set(trainer.logged_metrics.keys())
    assert expected_logged_metrics == logged_metrics

    # we don't want to enable val metrics during steps because it is not something that users should do
    # on purpose DO NOT allow step_b... it's silly to monitor val step metrics
    callback_metrics = set(trainer.callback_metrics.keys())
    callback_metrics.remove('debug_epoch')
    expected_cb_metrics = {'a', 'a2', 'b', 'a_epoch', 'b_epoch', 'a_step'}
    assert expected_cb_metrics == callback_metrics


def test__validation_step__step_end__epoch_end__log(tmpdir):
    """
    Tests that validation_step can log
    """
    os.environ['PL_DEV_DEBUG'] = '1'

    class TestModel(DeterministicModel):
        def training_step(self, batch, batch_idx):
            acc = self.step(batch, batch_idx)
            acc = acc + batch_idx
            self.log('a', acc)
            self.log('b', acc, on_step=True, on_epoch=True)
            self.training_step_called = True
            return acc

        def validation_step(self, batch, batch_idx):
            acc = self.step(batch, batch_idx)
            acc = acc + batch_idx
            self.log('c', acc)
            self.log('d', acc, on_step=True, on_epoch=True)
            self.validation_step_called = True
            return acc

        def validation_step_end(self, acc):
            self.validation_step_end_called = True
            self.log('e', acc)
            self.log('f', acc, on_step=True, on_epoch=True)
            return ['random_thing']

        def validation_epoch_end(self, outputs):
            self.log('g', torch.tensor(2, device=self.device), on_epoch=True)
            self.validation_epoch_end_called = True

        def backward(self, loss, optimizer, optimizer_idx):
            loss.backward()

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

    # make sure all the metrics are available for callbacks
    logged_metrics = set(trainer.logged_metrics.keys())
    expected_logged_metrics = {
        'epoch',
        'a',
        'b_step',
        'b_epoch',
        'c',
        'd_step/epoch_0',
        'd_step/epoch_1',
        'd_epoch',
        'e',
        'f_step/epoch_0',
        'f_step/epoch_1',
        'f_epoch',
        'g',
    }
    assert expected_logged_metrics == logged_metrics

    progress_bar_metrics = set(trainer.progress_bar_metrics.keys())
    expected_pbar_metrics = set()
    assert expected_pbar_metrics == progress_bar_metrics

    # we don't want to enable val metrics during steps because it is not something that users should do
    callback_metrics = set(trainer.callback_metrics.keys())
    callback_metrics.remove('debug_epoch')
    expected_cb_metrics = {'a', 'b', 'c', 'd', 'e', 'b_epoch', 'd_epoch', 'f_epoch', 'f', 'g', 'b_step'}
    assert expected_cb_metrics == callback_metrics


@pytest.mark.parametrize(['batches', 'log_interval', 'max_epochs'], [(1, 1, 1), (64, 32, 2)])
def test_eval_epoch_logging(tmpdir, batches, log_interval, max_epochs):
    """
    Tests that only training_step can be used
    """
    os.environ['PL_DEV_DEBUG'] = '1'

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

    # make sure all the metrics are available for callbacks
    logged_metrics = set(trainer.logged_metrics.keys())
    expected_logged_metrics = {
        'c',
        'd/e/f',
    }
    assert logged_metrics == expected_logged_metrics

    pbar_metrics = set(trainer.progress_bar_metrics.keys())
    expected_pbar_metrics = {'c'}
    assert pbar_metrics == expected_pbar_metrics

    callback_metrics = set(trainer.callback_metrics.keys())
    expected_callback_metrics = set()
    expected_callback_metrics = expected_callback_metrics.union(logged_metrics)
    expected_callback_metrics = expected_callback_metrics.union(pbar_metrics)
    callback_metrics.remove('debug_epoch')
    assert callback_metrics == expected_callback_metrics

    # assert the loggers received the expected number
    assert len(trainer.dev_debugger.logged_metrics) == max_epochs


def test_eval_float_logging(tmpdir):
    """
    Tests that only training_step can be used
    """
    os.environ['PL_DEV_DEBUG'] = '1'

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

    # make sure all the metrics are available for callbacks
    logged_metrics = set(trainer.logged_metrics.keys())
    expected_logged_metrics = {
        'a',
    }
    assert logged_metrics == expected_logged_metrics


def test_eval_logging_auto_reduce(tmpdir):
    """
    Tests that only training_step can be used
    """
    seed_everything(1234)

    os.environ['PL_DEV_DEBUG'] = '1'

    class TestModel(BoringModel):
        def on_pretrain_routine_end(self) -> None:
            self.seen_vals = []
            self.manual_epoch_end_mean = None

        def on_validation_epoch_start(self) -> None:
            self.seen_vals = []

        def validation_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            self.seen_vals.append(loss)
            self.log('val_loss', loss, on_epoch=True, on_step=True, prog_bar=True)
            return {"x": loss}

        def validation_epoch_end(self, outputs) -> None:
            for passed_in, manually_tracked in zip(outputs, self.seen_vals):
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
        checkpoint_callback=callbacks.ModelCheckpoint('val_loss')
    )
    trainer.fit(model)

    # make sure all the metrics are available for callbacks
    manual_mean = model.manual_epoch_end_mean
    callback_metrics = set(trainer.callback_metrics.keys())
    assert callback_metrics == {'debug_epoch', 'val_loss', 'val_loss_epoch'}

    # make sure values are correct
    assert trainer.logged_metrics['val_loss_epoch'] == manual_mean
    assert trainer.callback_metrics['val_loss'] == trainer.logged_metrics['val_loss_step/epoch_0']

    # make sure correct values were logged
    logged_val = trainer.dev_debugger.logged_metrics

    # sanity check
    assert logged_val[0]['global_step'] == 0
    assert logged_val[1]['global_step'] == 0

    # 3 val batches
    assert logged_val[2]['val_loss_step/epoch_0'] == model.seen_vals[0]
    assert logged_val[3]['val_loss_step/epoch_0'] == model.seen_vals[1]
    assert logged_val[4]['val_loss_step/epoch_0'] == model.seen_vals[2]

    # epoch mean
    assert logged_val[5]['val_loss_epoch'] == model.manual_epoch_end_mean

    # only those logged
    assert len(logged_val) == 6


def test_monitor_val_epoch_end(tmpdir):
    epoch_min_loss_override = 0
    model = SimpleModule()
    checkpoint_callback = callbacks.ModelCheckpoint(save_top_k=1, monitor="avg_val_loss")
    trainer = Trainer(
        max_epochs=epoch_min_loss_override + 2,
        logger=False,
        checkpoint_callback=checkpoint_callback,
    )
    trainer.fit(model)

def test_log_works_in_validation_callback(tmpdir):
    """
    Tests that log can be called within callback
    """
    import pytorch_lightning as pl

    os.environ['PL_DEV_DEBUG'] = '1'

    funcs_name = [f for f in dir(callbacks.Callback) if ("val" in f)]

    def logic_func(x):
        func_idx, func_name, args = x
        return f"""pl_module = locals().get('pl_module')\n  try:\n      pl_module.log('{func_name}', {func_idx})\n  except Exception as e:\n        print('{func_name}', e)""" \
            if "pl_module" in args else "pass"
    
    scripted_callback = create_scriptable_callback(funcs_name, logic_func)

    loss_values, patience, expected_stop_epoch = ([6, 5, 5, 5, 5, 5], 3, 4)

    class TestModel(BoringModel):

        validation_return_values = torch.Tensor(loss_values)
        count = 0

        def validation_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            self.log('c', self.count)
            return {"x": loss}

        def validation_epoch_end(self, outputs):

            loss = self.validation_return_values[self.count]
            self.count += 1
            return {"test_val_loss": loss}

    max_epochs = 2
    model = TestModel()
    early_stop_callback = pl.callbacks.EarlyStopping(monitor="test_val_loss", patience=3, verbose=True)

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=0,
        limit_val_batches=4,
        limit_test_batches=0,
        val_check_interval=1.0,
        num_sanity_val_steps=0,
        max_epochs=max_epochs,
        callbacks=[early_stop_callback, scripted_callback]
    )
    trainer.fit(model)

    expected_logged_metrics = set([f for f in funcs_name if f not in ["on_validation_epoch_start", "on_validation_start"]] + \
               [f"on_validation_epoch_start/epoch_{e}" for e in range(max_epochs)] + \
               [f"on_validation_start/epoch_{e}" for e in range(max_epochs)] + \
               ["c"])
    logged_metrics = set(trainer.logged_metrics.keys())
    assert logged_metrics == expected_logged_metrics, logged_metrics
