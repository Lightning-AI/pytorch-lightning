# Copyright The Lightning AI team.
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
import operator
from functools import partial
from unittest import mock
from unittest.mock import Mock

import pytest
import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.demos.boring_classes import BoringModel, RandomDataset
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.trainer.connectors.logger_connector.fx_validator import _FxValidator
from lightning.pytorch.trainer.connectors.logger_connector.result import _ResultCollection
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.imports import _TORCHMETRICS_GREATER_EQUAL_0_9_1
from lightning.pytorch.utilities.imports import _TORCHMETRICS_GREATER_EQUAL_0_11 as _TM_GE_0_11
from lightning_utilities.core.imports import compare_version
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, MeanAbsoluteError, MeanSquaredError, MetricCollection
from torchmetrics import AveragePrecision as AvgPre

from tests_pytorch.models.test_hooks import get_members


def test_fx_validator():
    funcs_name = get_members(Callback)

    callbacks_func = {
        "on_before_backward",
        "on_after_backward",
        "on_before_optimizer_step",
        "on_before_zero_grad",
        "on_fit_end",
        "on_fit_start",
        "on_exception",
        "on_load_checkpoint",
        "load_state_dict",
        "on_sanity_check_end",
        "on_sanity_check_start",
        "state_dict",
        "on_save_checkpoint",
        "on_test_batch_end",
        "on_test_batch_start",
        "on_test_end",
        "on_test_epoch_end",
        "on_test_epoch_start",
        "on_test_start",
        "on_train_batch_end",
        "on_train_batch_start",
        "on_train_end",
        "on_train_epoch_end",
        "on_train_epoch_start",
        "on_train_start",
        "on_validation_batch_end",
        "on_validation_batch_start",
        "on_validation_end",
        "on_validation_epoch_end",
        "on_validation_epoch_start",
        "on_validation_start",
        "on_predict_batch_end",
        "on_predict_batch_start",
        "on_predict_end",
        "on_predict_epoch_end",
        "on_predict_epoch_start",
        "on_predict_start",
        "setup",
        "teardown",
    }

    not_supported = {
        "on_fit_end",
        "on_fit_start",
        "on_exception",
        "on_load_checkpoint",
        "load_state_dict",
        "on_sanity_check_end",
        "on_sanity_check_start",
        "on_predict_batch_end",
        "on_predict_batch_start",
        "on_predict_end",
        "on_predict_epoch_end",
        "on_predict_epoch_start",
        "on_predict_start",
        "state_dict",
        "on_save_checkpoint",
        "on_test_end",
        "on_train_end",
        "on_validation_end",
        "setup",
        "teardown",
    }

    # Detected new callback function. Need to add its logging permission to FxValidator and update this test
    assert funcs_name == callbacks_func

    validator = _FxValidator()

    for func_name in funcs_name:
        # This summarizes where and what is currently possible to log using `self.log`
        is_stage = "train" in func_name or "test" in func_name or "validation" in func_name
        is_start = "start" in func_name or "batch" in func_name
        is_epoch = "epoch" in func_name
        on_step = is_stage and not is_start and not is_epoch
        on_epoch = True
        # creating allowed condition
        allowed = (
            is_stage
            or "batch" in func_name
            or "epoch" in func_name
            or "grad" in func_name
            or "backward" in func_name
            or "optimizer_step" in func_name
        )
        allowed = (
            allowed
            and "pretrain" not in func_name
            and "predict" not in func_name
            and func_name not in ["on_train_end", "on_test_end", "on_validation_end"]
        )
        if allowed:
            validator.check_logging_levels(fx_name=func_name, on_step=on_step, on_epoch=on_epoch)
            if not is_start and is_stage:
                with pytest.raises(MisconfigurationException, match="must be one of"):
                    validator.check_logging_levels(fx_name=func_name, on_step=True, on_epoch=on_epoch)
        else:
            assert func_name in not_supported
            with pytest.raises(MisconfigurationException, match="You can't"):
                validator.check_logging(fx_name=func_name)

    with pytest.raises(RuntimeError, match="Logging inside `foo` is not implemented"):
        validator.check_logging("foo")


class HookedCallback(Callback):
    def __init__(self, not_supported):
        def call(hook, trainer=None, model=None, *_, **__):
            if trainer is None:
                # `state_dict`, `load_state_dict` do not have the `Trainer` available
                assert hook in ("state_dict", "load_state_dict")
                return

            lightning_module = trainer.lightning_module or model

            if hook in not_supported:
                with pytest.raises(MisconfigurationException, match=not_supported[hook]):
                    lightning_module.log("anything", 1)
            else:
                lightning_module.log(hook, 1)

        for h in get_members(Callback):
            setattr(self, h, partial(call, h))


class HookedModel(BoringModel):
    def __init__(self, not_supported):
        super().__init__()
        pl_module_hooks = get_members(LightningModule)
        pl_module_hooks.difference_update({"log", "log_dict"})
        pl_module_hooks.discard("configure_sharded_model")
        # remove `nn.Module` hooks
        module_hooks = get_members(torch.nn.Module)
        pl_module_hooks.difference_update(module_hooks)

        def call(hook, fn, *args, **kwargs):
            out = fn(*args, **kwargs)

            if hook in not_supported:
                with pytest.raises(MisconfigurationException, match=not_supported[hook]):
                    self.log("anything", 1)
            else:
                self.log(hook, 1)
            return out

        for h in pl_module_hooks:
            attr = getattr(self, h)
            setattr(self, h, partial(call, h, attr))


def test_fx_validator_integration(tmp_path):
    """Tries to log inside all `LightningModule` and `Callback` hooks to check any expected errors."""
    not_supported = {
        None: "`self.trainer` reference is not registered",
        "setup": "You can't",
        "configure_model": "You can't",
        "configure_optimizers": "You can't",
        "on_fit_start": "You can't",
        "train_dataloader": "You can't",
        "val_dataloader": "You can't",
        "on_before_batch_transfer": "You can't",
        "transfer_batch_to_device": "You can't",
        "on_after_batch_transfer": "You can't",
        "on_validation_end": "You can't",
        "on_train_end": "You can't",
        "on_fit_end": "You can't",
        "teardown": "You can't",
        "on_sanity_check_start": "You can't",
        "on_sanity_check_end": "You can't",
        "prepare_data": "You can't",
        "configure_callbacks": "You can't",
        "on_validation_model_zero_grad": "You can't",
        "on_validation_model_eval": "You can't",
        "on_validation_model_train": "You can't",
        "lr_scheduler_step": "You can't",
        "on_save_checkpoint": "You can't",
        "on_load_checkpoint": "You can't",
        "on_exception": "You can't",
    }
    model = HookedModel(not_supported)

    with pytest.warns(UserWarning, match=not_supported[None]):
        model.log("foo", 1)

    callback = HookedCallback(not_supported)
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=2,
        limit_train_batches=1,
        limit_val_batches=1,
        limit_test_batches=1,
        limit_predict_batches=1,
        callbacks=callback,
    )
    trainer.fit(model)

    not_supported.update({
        # `lightning_module` ref is now present from the `fit` call
        "test_dataloader": "You can't",
        "on_test_model_eval": "You can't",
        "on_test_model_train": "You can't",
        "on_test_end": "You can't",
    })
    trainer.test(model, verbose=False)

    not_supported.update({k: "result collection is not registered yet" for k in not_supported})
    not_supported.update({
        "predict_dataloader": "result collection is not registered yet",
        "on_predict_model_eval": "result collection is not registered yet",
        "on_predict_start": "result collection is not registered yet",
        "on_predict_epoch_start": "result collection is not registered yet",
        "on_predict_batch_start": "result collection is not registered yet",
        "predict_step": "result collection is not registered yet",
        "on_predict_batch_end": "result collection is not registered yet",
        "on_predict_epoch_end": "result collection is not registered yet",
        "on_predict_end": "result collection is not registered yet",
    })
    trainer.predict(model)


@pytest.mark.parametrize("add_dataloader_idx", [False, True])
def test_auto_add_dataloader_idx(tmp_path, add_dataloader_idx):
    """Test that auto_add_dataloader_idx argument works."""

    class TestModel(BoringModel):
        def val_dataloader(self):
            dl = super().val_dataloader()
            return [dl, dl]

        def validation_step(self, *args, **kwargs):
            output = super().validation_step(*args[:-1], **kwargs)
            name = "val_loss" if add_dataloader_idx else f"val_loss_custom_naming_{args[-1]}"

            self.log(name, output["x"], add_dataloader_idx=add_dataloader_idx)
            return output

    model = TestModel()

    trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=2)
    trainer.fit(model)
    logged = trainer.logged_metrics

    # Check that the correct keys exist
    if add_dataloader_idx:
        assert "val_loss/dataloader_idx_0" in logged
        assert "val_loss/dataloader_idx_1" in logged
    else:
        assert "val_loss_custom_naming_0" in logged
        assert "val_loss_custom_naming_1" in logged


def test_metrics_reset(tmp_path):
    """Tests that metrics are reset correctly after the end of the train/val/test epoch."""

    class TestModel(LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(32, 1)

        def _create_metrics(self):
            acc = Accuracy(task="binary") if _TM_GE_0_11 else Accuracy()
            acc.reset = mock.Mock(side_effect=acc.reset)
            ap = AvgPre(task="binary") if _TM_GE_0_11 else AvgPre(num_classes=1, pos_label=1)
            ap.reset = mock.Mock(side_effect=ap.reset)
            return acc, ap

        def setup(self, stage):
            fn = stage.value
            if fn == "fit":
                for stage in ("train", "validate"):
                    acc, ap = self._create_metrics()
                    self.add_module(f"acc_{fn}_{stage}", acc)
                    self.add_module(f"ap_{fn}_{stage}", ap)
            else:
                acc, ap = self._create_metrics()
                stage = self.trainer.state.stage.value
                self.add_module(f"acc_{fn}_{stage}", acc)
                self.add_module(f"ap_{fn}_{stage}", ap)

        def forward(self, x):
            return self.layer(x)

        def _step(self, batch):
            fn, stage = self.trainer.state.fn.value, self.trainer.state.stage.value

            logits = self(batch)
            loss = logits.sum()
            self.log(f"loss/{fn}_{stage}", loss)

            acc = self._modules[f"acc_{fn}_{stage}"]
            ap = self._modules[f"ap_{fn}_{stage}"]

            preds = torch.rand(len(batch))  # Fake preds
            labels = torch.randint(0, 1, [len(batch)])  # Fake targets
            acc(preds, labels)
            ap(preds, labels)

            # Metric.forward calls reset so reset the mocks here
            acc.reset.reset_mock()
            ap.reset.reset_mock()

            self.log(f"acc/{fn}_{stage}", acc)
            self.log(f"ap/{fn}_{stage}", ap)

            return loss

        def training_step(self, batch, batch_idx, *args, **kwargs):
            return self._step(batch)

        def validation_step(self, batch, batch_idx, *args, **kwargs):
            if self.trainer.sanity_checking:
                return None
            return self._step(batch)

        def test_step(self, batch, batch_idx, *args, **kwargs):
            return self._step(batch)

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

    def _assert_called(model, fn, stage):
        acc = model._modules[f"acc_{fn}_{stage}"]
        ap = model._modules[f"ap_{fn}_{stage}"]
        acc.reset.assert_called_once()
        ap.reset.assert_called_once()

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        max_epochs=1,
        enable_progress_bar=False,
        num_sanity_val_steps=2,
        enable_checkpointing=False,
    )

    trainer.fit(model)
    _assert_called(model, "fit", "train")
    _assert_called(model, "fit", "validate")

    trainer.validate(model)
    _assert_called(model, "validate", "validate")

    trainer.test(model)
    _assert_called(model, "test", "test")


@pytest.mark.skipif(
    compare_version("torchmetrics", operator.lt, "0.8.0"), reason="torchmetrics>=0.8.0 required for compute groups"
)
@pytest.mark.parametrize("compute_groups", [True, False])
def test_metriccollection_compute_groups(tmp_path, compute_groups):
    def assertion_calls(keep_base: bool, copy_state: bool):
        if _TORCHMETRICS_GREATER_EQUAL_0_9_1:
            assert copy_state != compute_groups

        assert not keep_base

    class CustomMetricsCollection(MetricCollection):
        wrapped_assertion_calls = Mock(wraps=assertion_calls)

        def items(self, keep_base: bool = False, copy_state: bool = True):
            if getattr(self, "_is_currently_logging", False):
                self.wrapped_assertion_calls(keep_base, copy_state)

            return super().items(keep_base=keep_base, copy_state=copy_state)

    class DummyModule(LightningModule):
        def __init__(self):
            super().__init__()
            if compare_version("torchmetrics", operator.ge, "0.10.0"):
                from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision

                metrics = [
                    MulticlassAccuracy(num_classes=10, average="micro"),
                    MulticlassPrecision(num_classes=10, average="micro"),
                ]
            else:
                from torchmetrics import Accuracy, Precision

                metrics = [Accuracy(num_classes=10, average="micro"), Precision(num_classes=10, average="micro")]

            self.metrics = CustomMetricsCollection(
                metrics,
                compute_groups=compute_groups,
            )
            self.layer = torch.nn.Linear(32, 10)

        def training_step(self, batch):
            self.metrics(torch.rand(10, 10).softmax(-1), torch.randint(0, 10, (10,)))
            self.metrics._is_currently_logging = True
            self.log_dict(self.metrics, on_step=True, on_epoch=True)
            self.metrics._is_currently_logging = False
            return self.layer(batch).sum()

        def train_dataloader(self):
            return DataLoader(RandomDataset(32, 64))

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.1)

        def on_train_epoch_end(self) -> None:
            self.metrics.wrapped_assertion_calls.call_count == 2
            self.metrics.wrapped_assertion_calls.reset_mock()

    trainer = Trainer(
        default_root_dir=tmp_path,
        limit_train_batches=2,
        limit_val_batches=0,
        max_epochs=1,
        enable_progress_bar=False,
        enable_checkpointing=False,
    )
    trainer.fit(DummyModule())


def test_result_collection_on_tensor_with_mean_reduction():
    result_collection = _ResultCollection(True)
    product = [(True, True), (False, True), (True, False), (False, False)]
    values = torch.arange(1, 10)
    batches = values * values

    for i, v in enumerate(values):
        for prog_bar in [False, True]:
            for logger in [False, True]:
                for on_step, on_epoch in product:
                    name = "loss"
                    if on_step:
                        name += "_on_step"
                    if on_epoch:
                        name += "_on_epoch"
                    if prog_bar:
                        name += "_prog_bar"
                    if logger:
                        name += "_logger"
                    log_kwargs = {
                        "fx": "training_step",
                        "name": name,
                        "value": v,
                        "on_step": on_step,
                        "on_epoch": on_epoch,
                        "batch_size": batches[i],
                        "prog_bar": prog_bar,
                        "logger": logger,
                    }
                    if not on_step and not on_epoch:
                        with pytest.raises(MisconfigurationException, match="on_step=False, on_epoch=False"):
                            result_collection.log(**log_kwargs)
                    else:
                        result_collection.log(**log_kwargs)

    total_value = sum(values * batches)
    total_batches = sum(batches)
    assert result_collection["training_step.loss_on_step_on_epoch"].value == total_value
    assert result_collection["training_step.loss_on_step_on_epoch"].cumulated_batch_size == total_batches

    batch_metrics = result_collection.metrics(True)
    max_ = max(values)
    assert batch_metrics["pbar"] == {
        "loss_on_step_on_epoch_prog_bar_step": max_,
        "loss_on_step_on_epoch_prog_bar_logger_step": max_,
        "loss_on_step_prog_bar": max_,
        "loss_on_step_prog_bar_logger": max_,
    }
    assert batch_metrics["log"] == {
        "loss_on_step_on_epoch_logger_step": max_,
        "loss_on_step_logger": max_,
        "loss_on_step_on_epoch_prog_bar_logger_step": max_,
        "loss_on_step_prog_bar_logger": max_,
    }
    assert batch_metrics["callback"] == {
        "loss_on_step": max_,
        "loss_on_step_logger": max_,
        "loss_on_step_on_epoch": max_,
        "loss_on_step_on_epoch_logger": max_,
        "loss_on_step_on_epoch_logger_step": max_,
        "loss_on_step_on_epoch_prog_bar": max_,
        "loss_on_step_on_epoch_prog_bar_logger": max_,
        "loss_on_step_on_epoch_prog_bar_logger_step": max_,
        "loss_on_step_on_epoch_prog_bar_step": max_,
        "loss_on_step_on_epoch_step": max_,
        "loss_on_step_prog_bar": max_,
        "loss_on_step_prog_bar_logger": max_,
    }

    epoch_metrics = result_collection.metrics(False)
    mean = total_value / total_batches
    assert epoch_metrics["pbar"] == {
        "loss_on_epoch_prog_bar": mean,
        "loss_on_epoch_prog_bar_logger": mean,
        "loss_on_step_on_epoch_prog_bar_epoch": mean,
        "loss_on_step_on_epoch_prog_bar_logger_epoch": mean,
    }
    assert epoch_metrics["log"] == {
        "loss_on_epoch_logger": mean,
        "loss_on_epoch_prog_bar_logger": mean,
        "loss_on_step_on_epoch_logger_epoch": mean,
        "loss_on_step_on_epoch_prog_bar_logger_epoch": mean,
    }
    assert epoch_metrics["callback"] == {
        "loss_on_epoch": mean,
        "loss_on_epoch_logger": mean,
        "loss_on_epoch_prog_bar": mean,
        "loss_on_epoch_prog_bar_logger": mean,
        "loss_on_step_on_epoch": mean,
        "loss_on_step_on_epoch_epoch": mean,
        "loss_on_step_on_epoch_logger": mean,
        "loss_on_step_on_epoch_logger_epoch": mean,
        "loss_on_step_on_epoch_prog_bar": mean,
        "loss_on_step_on_epoch_prog_bar_epoch": mean,
        "loss_on_step_on_epoch_prog_bar_logger": mean,
        "loss_on_step_on_epoch_prog_bar_logger_epoch": mean,
    }


@pytest.mark.parametrize("logger", [False, True])
def test_logged_metrics_has_logged_epoch_value(tmp_path, logger):
    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx):
            self.log("epoch", -batch_idx, logger=True)
            return super().training_step(batch, batch_idx)

    model = TestModel()
    trainer_kwargs = {
        "default_root_dir": tmp_path,
        "limit_train_batches": 2,
        "limit_val_batches": 0,
        "max_epochs": 1,
        "logger": False,
    }
    if logger:
        trainer_kwargs["logger"] = CSVLogger(tmp_path)
    trainer = Trainer(**trainer_kwargs)
    if not logger:
        with pytest.warns(match=r"log\('epoch', ..., logger=True\)` but have no logger"):
            trainer.fit(model)
    else:
        trainer.fit(model)

    # should not get overridden if logged manually
    assert trainer.logged_metrics == {"epoch": -1}


def test_result_collection_batch_size_extraction():
    fx_name = "training_step"
    log_val = torch.tensor(7.0)

    results = _ResultCollection(training=True)
    results.batch = torch.randn(1, 4)
    train_mse = MeanSquaredError()
    train_mse(torch.randn(4, 5), torch.randn(4, 5))
    results.log(fx_name, "mse", train_mse, on_step=False, on_epoch=True)
    results.log(fx_name, "log_val", log_val, on_step=False, on_epoch=True)
    assert results.batch_size == 1
    assert isinstance(results["training_step.mse"].value, MeanSquaredError)
    assert results["training_step.log_val"].value == log_val

    results = _ResultCollection(training=True)
    results.batch = torch.randn(1, 4)
    results.log(fx_name, "train_log", log_val, on_step=False, on_epoch=True)
    assert results.batch_size == 1
    assert results["training_step.train_log"].value == log_val
    assert results["training_step.train_log"].cumulated_batch_size == 1


def test_result_collection_no_batch_size_extraction():
    results = _ResultCollection(training=True)
    results.batch = torch.randn(1, 4)
    fx_name = "training_step"
    batch_size = 10
    log_val = torch.tensor(7.0)

    train_mae = MeanAbsoluteError()
    train_mae(torch.randn(4, 5), torch.randn(4, 5))
    results.log(fx_name, "step_log_val", log_val, on_step=True, on_epoch=False)
    results.log(fx_name, "epoch_log_val", log_val, on_step=False, on_epoch=True, batch_size=batch_size)
    results.log(fx_name, "epoch_sum_log_val", log_val, on_step=True, on_epoch=True, reduce_fx="sum")
    results.log(fx_name, "train_mae", train_mae, on_step=True, on_epoch=False)

    assert results.batch_size is None
    assert isinstance(results["training_step.train_mae"].value, MeanAbsoluteError)
    assert results["training_step.step_log_val"].value == log_val
    assert results["training_step.step_log_val"].cumulated_batch_size == 0
    assert results["training_step.epoch_log_val"].value == log_val * batch_size
    assert results["training_step.epoch_log_val"].cumulated_batch_size == batch_size
    assert results["training_step.epoch_sum_log_val"].value == log_val
