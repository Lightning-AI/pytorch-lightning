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
import os
from copy import deepcopy
from typing import Any, Callable
from unittest import mock

import pytest
import torch
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, AveragePrecision

from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.core.step_result import Result
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.trainer.connectors.logger_connector.callback_hook_validator import CallbackHookNameValidator
from pytorch_lightning.trainer.connectors.logger_connector.metrics_holder import MetricsHolder
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers.boring_model import BoringModel, RandomDataset
from tests.helpers.runif import RunIf


def decorator_with_arguments(fx_name: str = '', hook_fx_name: str = None) -> Callable:

    def decorator(func: Callable) -> Callable:

        def wrapper(self, *args, **kwargs) -> Any:
            # Set information
            self._current_fx_name = fx_name
            self._current_hook_fx_name = hook_fx_name
            self._results = Result()

            result = func(self, *args, **kwargs)

            # cache metrics
            self.trainer.logger_connector.cache_logged_metrics()
            return result

        return wrapper

    return decorator


def test__logger_connector__epoch_result_store__train(tmpdir):
    """
    Tests that LoggerConnector will properly capture logged information
    and reduce them
    """

    class TestModel(BoringModel):

        train_losses = []

        @decorator_with_arguments(fx_name="training_step")
        def training_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)

            self.train_losses.append(loss)

            self.log("train_loss", loss, on_step=True, on_epoch=True)

            return {"loss": loss}

        def training_step_end(self, *_):
            self.train_results = deepcopy(self.trainer.logger_connector.cached_results)

    model = TestModel()
    model.training_epoch_end = None
    model.val_dataloader = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=4,
        max_epochs=1,
        log_every_n_steps=1,
        weights_summary=None,
    )
    trainer.fit(model)

    train_results = model.train_results

    assert len(train_results(fx_name="training_step", dl_idx=0, opt_idx=0)) == 2
    generated = train_results(fx_name="training_step", dl_idx=0, opt_idx=0, batch_idx=0, split_idx=0)["train_loss"]
    assert generated == model.train_losses[0]
    generated = train_results(fx_name="training_step", dl_idx=0, opt_idx=0, batch_idx=1, split_idx=0)["train_loss"]
    assert generated == model.train_losses[1]

    assert train_results.has_reduced is not True

    train_results.has_batch_loop_finished = True

    assert train_results.has_reduced is True

    generated = train_results(fx_name="training_step", dl_idx=0, opt_idx=0, reduced=True)['train_loss_epoch'].item()
    excepted = torch.stack(model.train_losses).mean().item()
    assert generated == excepted


def test__logger_connector__epoch_result_store__train__tbptt(tmpdir):
    """
    Tests that LoggerConnector will properly capture logged information with ttbt
    and reduce them
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

        train_losses = []

        def __init__(self):
            super().__init__()
            self.test_hidden = None
            self.layer = torch.nn.Linear(2, 2)

        @decorator_with_arguments(fx_name="training_step")
        def training_step(self, batch, batch_idx, hiddens):
            assert hiddens == self.test_hidden, "Hidden state not persistent between tbptt steps"
            self.test_hidden = torch.rand(1)

            x_tensor, y_list = batch
            assert x_tensor.shape[1] == truncated_bptt_steps, "tbptt split Tensor failed"

            y_tensor = torch.tensor(y_list, dtype=x_tensor.dtype)
            assert y_tensor.shape[1] == truncated_bptt_steps, "tbptt split list failed"

            pred = self(x_tensor.view(batch_size, truncated_bptt_steps))
            loss = torch.nn.functional.mse_loss(pred, y_tensor.view(batch_size, truncated_bptt_steps))

            self.train_losses.append(loss)

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

        def training_step_end(self, training_step_output):
            self.train_results = deepcopy(self.trainer.logger_connector.cached_results)
            # must return
            return training_step_output

    model = TestModel()
    model.training_epoch_end = None
    model.example_input_array = torch.randn(5, truncated_bptt_steps)

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=10,
        limit_val_batches=0,
        truncated_bptt_steps=truncated_bptt_steps,
        max_epochs=1,
        log_every_n_steps=1,
        weights_summary=None,
    )
    trainer.fit(model)

    train_results = model.train_results

    generated = train_results(fx_name="training_step", dl_idx=0, opt_idx=0, batch_idx=0)
    assert len(generated) == len(model.train_losses)

    # assert reduction didn't happen yet
    assert train_results.has_reduced is False

    # Launch reduction
    train_results.has_batch_loop_finished = True

    # assert reduction did happen
    assert train_results.has_reduced is True

    generated = train_results(fx_name="training_step", dl_idx=0, opt_idx=0, reduced=True)['a_epoch'].item()
    assert generated == torch.stack(model.train_losses).mean().item()


@pytest.mark.parametrize('num_dataloaders', [1, 2])
def test__logger_connector__epoch_result_store__test_multi_dataloaders(tmpdir, num_dataloaders):
    """
    Tests that LoggerConnector will properly capture logged information in multi dataloaders scenario
    """

    class TestModel(BoringModel):
        test_losses = {dl_idx: [] for dl_idx in range(num_dataloaders)}

        @decorator_with_arguments(fx_name="test_step")
        def test_step(self, batch, batch_idx, dl_idx=0):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            self.test_losses[dl_idx].append(loss)
            self.log("test_loss", loss, on_step=True, on_epoch=True)
            return {"test_loss": loss}

        def on_test_batch_end(self, *args, **kwargs):
            # save objects as it will be reset at the end of epoch.
            self.batch_results = deepcopy(self.trainer.logger_connector.cached_results)

        def on_test_epoch_end(self):
            # save objects as it will be reset at the end of epoch.
            self.reduce_results = deepcopy(self.trainer.logger_connector.cached_results)

        def test_dataloader(self):
            return [super().test_dataloader()] * num_dataloaders

    model = TestModel()
    model.test_epoch_end = None
    limit_test_batches = 4

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=0,
        limit_val_batches=0,
        limit_test_batches=limit_test_batches,
        max_epochs=1,
        log_every_n_steps=1,
        weights_summary=None,
    )
    trainer.test(model)

    test_results = model.batch_results

    generated = test_results(fx_name="test_step")
    assert len(generated) == num_dataloaders

    for dl_idx in range(num_dataloaders):
        generated = test_results(fx_name="test_step", dl_idx=dl_idx)
        assert len(generated) == limit_test_batches

    test_results = model.reduce_results

    for dl_idx in range(num_dataloaders):
        expected = torch.stack(model.test_losses[dl_idx]).mean()
        generated = test_results(fx_name="test_step", dl_idx=dl_idx, reduced=True)["test_loss_epoch"]
        torch.testing.assert_allclose(generated, expected)


def test_call_back_validator(tmpdir):

    funcs_name = sorted([f for f in dir(Callback) if not f.startswith('_')])

    callbacks_func = [
        'on_after_backward',
        'on_batch_end',
        'on_batch_start',
        'on_before_accelerator_backend_setup',
        'on_before_zero_grad',
        'on_epoch_end',
        'on_epoch_start',
        'on_fit_end',
        'on_configure_sharded_model',
        'on_fit_start',
        'on_init_end',
        'on_init_start',
        'on_keyboard_interrupt',
        'on_load_checkpoint',
        'on_pretrain_routine_end',
        'on_pretrain_routine_start',
        'on_sanity_check_end',
        'on_sanity_check_start',
        'on_save_checkpoint',
        'on_test_batch_end',
        'on_test_batch_start',
        'on_test_end',
        'on_test_epoch_end',
        'on_test_epoch_start',
        'on_test_start',
        'on_train_batch_end',
        'on_train_batch_start',
        'on_train_end',
        'on_train_epoch_end',
        'on_train_epoch_start',
        'on_train_start',
        'on_validation_batch_end',
        'on_validation_batch_start',
        'on_validation_end',
        'on_validation_epoch_end',
        'on_validation_epoch_start',
        'on_validation_start',
        'setup',
        'teardown',
    ]

    not_supported = [
        "on_before_accelerator_backend_setup",
        "on_fit_end",
        "on_fit_start",
        "on_configure_sharded_model",
        "on_init_end",
        "on_init_start",
        "on_keyboard_interrupt",
        "on_load_checkpoint",
        "on_pretrain_routine_end",
        "on_pretrain_routine_start",
        "on_sanity_check_end",
        "on_sanity_check_start",
        "on_save_checkpoint",
        "on_test_end",
        "on_train_end",
        "on_validation_end",
        "setup",
        "teardown",
    ]

    assert (
        funcs_name == sorted(callbacks_func)
    ), """Detected new callback function.
        Need to add its logging permission to CallbackHookNameValidator and update this test"""

    validator = CallbackHookNameValidator()

    for func_name in funcs_name:
        # This summarizes where and what is currently possible to log using `self.log`
        is_stage = "train" in func_name or "test" in func_name or "validation" in func_name
        is_start = "start" in func_name or "batch" in func_name
        on_step = is_stage and is_start
        on_epoch = True
        # creating allowed condition
        allowed = (
            is_stage or "batch" in func_name or "epoch" in func_name or "grad" in func_name or "backward" in func_name
        )
        allowed = (
            allowed and "pretrain" not in func_name
            and func_name not in ["on_train_end", "on_test_end", "on_validation_end"]
        )
        if allowed:
            validator.check_logging_in_callbacks(current_hook_fx_name=func_name, on_step=on_step, on_epoch=on_epoch)
            if not is_start and is_stage:
                with pytest.raises(MisconfigurationException, match="function supports only"):
                    validator.check_logging_in_callbacks(
                        current_hook_fx_name=func_name, on_step=True, on_epoch=on_epoch
                    )
        else:
            assert func_name in not_supported
            with pytest.raises(MisconfigurationException, match="function doesn't support"):
                validator.check_logging_in_callbacks(current_hook_fx_name=func_name, on_step=on_step, on_epoch=on_epoch)

        # should not fail
        validator.check_logging_in_callbacks(current_hook_fx_name=None, on_step=None, on_epoch=None)


@RunIf(min_gpus=2)
def test_epoch_results_cache_dp(tmpdir):

    root_device = torch.device("cuda", 0)

    class TestModel(BoringModel):

        def training_step(self, *args, **kwargs):
            result = super().training_step(*args, **kwargs)
            self.log("train_loss_epoch", result["loss"], on_step=False, on_epoch=True)
            return result

        def training_step_end(self, training_step_outputs):  # required for dp
            loss = training_step_outputs["loss"].mean()
            return loss

        def training_epoch_end(self, outputs):
            assert all(out["loss"].device == root_device for out in outputs)
            assert self.trainer.callback_metrics["train_loss_epoch"].device == root_device

        def validation_step(self, *args, **kwargs):
            val_loss = torch.rand(1, device=torch.device("cuda", 1))
            self.log("val_loss_epoch", val_loss, on_step=False, on_epoch=True)
            return val_loss

        def validation_epoch_end(self, outputs):
            assert all(loss.device == root_device for loss in outputs)
            assert self.trainer.callback_metrics["val_loss_epoch"].device == root_device

        def test_step(self, *args, **kwargs):
            test_loss = torch.rand(1, device=torch.device("cuda", 1))
            self.log("test_loss_epoch", test_loss, on_step=False, on_epoch=True)
            return test_loss

        def test_epoch_end(self, outputs):
            assert all(loss.device == root_device for loss in outputs)
            assert self.trainer.callback_metrics["test_loss_epoch"].device == root_device

        def train_dataloader(self):
            return DataLoader(RandomDataset(32, 64), batch_size=4)

        def val_dataloader(self):
            return DataLoader(RandomDataset(32, 64), batch_size=4)

        def test_dataloader(self):
            return DataLoader(RandomDataset(32, 64), batch_size=4)

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator="dp",
        gpus=2,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=1,
    )
    trainer.fit(model)
    trainer.test(model, ckpt_path=None)


@pytest.mark.parametrize('to_float', [False, True])
def test_metrics_holder(to_float, tmpdir):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    preds = torch.tensor([[0.9, 0.1]], device=device)

    def is_float(value: Any) -> bool:
        return isinstance(value, float)

    excepted_function = is_float if to_float else torch.is_tensor
    targets = torch.tensor([1], device=device)
    acc = Accuracy().to(device)
    metric_holder = MetricsHolder(to_float=to_float)
    metric_holder.update({
        "x": 1,
        "y": torch.tensor(2),
        "z": acc(preds, targets),
    })
    metric_holder.convert(device)
    metrics = metric_holder.metrics
    assert excepted_function(metrics["x"])
    assert excepted_function(metrics["y"])
    assert excepted_function(metrics["z"])


def test_metric_holder_raises(tmpdir):
    """Check that an error is raised when trying to convert non-scalar tensors"""

    class TestModel(BoringModel):

        def validation_step(self, batch, *args, **kwargs):
            output = self(batch)
            self.log('test', output)

        def test_step(self, *args, **kwargs):
            return self.validation_step(*args, **kwargs)

    model = TestModel()
    model.validation_epoch_end = None
    model.test_epoch_end = None

    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)

    match = "The metric `test` does not contain a single element"
    with pytest.raises(MisconfigurationException, match=match):
        trainer.validate(model)
    with pytest.raises(MisconfigurationException, match=match):
        trainer.test(model)


def test_can_return_tensor_with_more_than_one_element(tmpdir):
    """Ensure {validation,test}_step return values are not included as callback metrics. #6623"""

    class TestModel(BoringModel):

        def validation_step(self, batch, *args, **kwargs):
            return {"val": torch.tensor([0, 1])}

        def validation_epoch_end(self, outputs):
            # ensure validation step returns still appear here
            assert len(outputs) == 2
            assert all(list(d) == ["val"] for d in outputs)  # check keys
            assert all(torch.equal(d["val"], torch.tensor([0, 1])) for d in outputs)  # check values

        def test_step(self, batch, *args, **kwargs):
            return {"test": torch.tensor([0, 1])}

        def test_epoch_end(self, outputs):
            assert len(outputs) == 2
            assert all(list(d) == ["test"] for d in outputs)  # check keys
            assert all(torch.equal(d["test"], torch.tensor([0, 1])) for d in outputs)  # check values

    model = TestModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=2, progress_bar_refresh_rate=0)
    trainer.fit(model)
    trainer.validate(model)
    trainer.test(model)


def test_logging_to_progress_bar_with_reserved_key(tmpdir):
    """ Test that logging a metric with a reserved name to the progress bar raises a warning. """

    class TestModel(BoringModel):

        def training_step(self, *args, **kwargs):
            output = super().training_step(*args, **kwargs)
            self.log("loss", output["loss"], prog_bar=True)
            return output

    model = TestModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    with pytest.warns(UserWarning, match="The progress bar already tracks a metric with the .* 'loss'"):
        trainer.fit(model)


@pytest.mark.parametrize("add_dataloader_idx", [False, True])
def test_auto_add_dataloader_idx(tmpdir, add_dataloader_idx):
    """ test that auto_add_dataloader_idx argument works """

    class TestModel(BoringModel):

        def val_dataloader(self):
            dl = super().val_dataloader()
            return [dl, dl]

        def validation_step(self, *args, **kwargs):
            output = super().validation_step(*args[:-1], **kwargs)
            if add_dataloader_idx:
                name = "val_loss"
            else:
                name = f"val_loss_custom_naming_{args[-1]}"

            self.log(name, output["x"], add_dataloader_idx=add_dataloader_idx)
            return output

    model = TestModel()
    model.validation_epoch_end = None

    trainer = Trainer(default_root_dir=tmpdir, max_steps=5)
    trainer.fit(model)
    logged = trainer.logged_metrics

    # Check that the correct keys exist
    if add_dataloader_idx:
        assert 'val_loss/dataloader_idx_0' in logged
        assert 'val_loss/dataloader_idx_1' in logged
    else:
        assert 'val_loss_custom_naming_0' in logged
        assert 'val_loss_custom_naming_1' in logged


@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
def test_logged_metrics_steps(tmpdir):

    class TestModel(BoringModel):

        def validation_step(self, batch, batch_idx):
            loss_val = torch.randn(1)
            self.log('val_loss', loss_val)
            return loss_val

    model = TestModel()
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

    assert trainer.dev_debugger.logged_metrics[0]['global_step'] == 1
    assert trainer.dev_debugger.logged_metrics[1]['global_step'] == 3


def test_metrics_reset(tmpdir):
    """Tests that metrics are reset correctly after the end of the train/val/test epoch."""

    class TestModel(LightningModule):

        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(32, 1)

            for stage in ['train', 'val', 'test']:
                acc = Accuracy()
                acc.reset = mock.Mock(side_effect=acc.reset)
                ap = AveragePrecision(num_classes=1, pos_label=1)
                ap.reset = mock.Mock(side_effect=ap.reset)
                self.add_module(f"acc_{stage}", acc)
                self.add_module(f"ap_{stage}", ap)

        def forward(self, x):
            return self.layer(x)

        def _step(self, stage, batch):
            labels = (batch.detach().sum(1) > 0).float()  # Fake some targets
            logits = self.forward(batch)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.unsqueeze(1))
            probs = torch.sigmoid(logits.detach())
            self.log(f"loss/{stage}", loss)

            acc = self._modules[f"acc_{stage}"]
            ap = self._modules[f"ap_{stage}"]

            labels_int = labels.to(torch.long)
            acc(probs, labels_int)
            ap(probs, labels_int)

            # Metric.forward calls reset so reset the mocks here
            acc.reset.reset_mock()
            ap.reset.reset_mock()

            self.log(f"{stage}/accuracy", acc)
            self.log(f"{stage}/ap", ap)

            return loss

        def training_step(self, batch, batch_idx, *args, **kwargs):
            return self._step('train', batch)

        def validation_step(self, batch, batch_idx, *args, **kwargs):
            return self._step('val', batch)

        def test_step(self, batch, batch_idx, *args, **kwargs):
            return self._step('test', batch)

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            return [optimizer], [lr_scheduler]

        def train_dataloader(self):
            return DataLoader(RandomDataset(32, 64))

        def val_dataloader(self):
            return DataLoader(RandomDataset(32, 64))

        def test_dataloader(self):
            return DataLoader(RandomDataset(32, 64))

        def _assert_epoch_end(self, stage):
            acc = self._modules[f"acc_{stage}"]
            ap = self._modules[f"ap_{stage}"]

            acc.reset.asset_not_called()
            ap.reset.assert_not_called()

        def on_train_epoch_end(self, outputs):
            self._assert_epoch_end('train')

        def on_validation_epoch_end(self, outputs):
            self._assert_epoch_end('val')

        def on_test_epoch_end(self, outputs):
            self._assert_epoch_end('test')

    def _assert_called(model, stage):
        acc = model._modules[f"acc_{stage}"]
        ap = model._modules[f"ap_{stage}"]

        acc.reset.assert_called_once()
        acc.reset.reset_mock()

        ap.reset.assert_called_once()
        ap.reset.reset_mock()

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        max_epochs=1,
        progress_bar_refresh_rate=0,
    )

    trainer.fit(model)
    _assert_called(model, 'train')
    _assert_called(model, 'val')

    trainer.validate(model)
    _assert_called(model, 'val')

    trainer.test(model)
    _assert_called(model, 'test')
