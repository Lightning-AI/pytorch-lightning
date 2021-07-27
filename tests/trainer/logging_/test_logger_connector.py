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
from unittest import mock

import pytest
import torch
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, AveragePrecision

from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.trainer.connectors.logger_connector.fx_validator import FxValidator
from pytorch_lightning.trainer.connectors.logger_connector.result import MetricSource, ResultCollection
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers.boring_model import BoringModel, RandomDataset
from tests.helpers.runif import RunIf


def test_fx_validator(tmpdir):
    funcs_name = sorted(f for f in dir(Callback) if not f.startswith("_"))

    callbacks_func = [
        "on_before_backward",
        "on_after_backward",
        "on_before_optimizer_step",
        "on_batch_end",
        "on_batch_start",
        "on_before_accelerator_backend_setup",
        "on_before_zero_grad",
        "on_epoch_end",
        "on_epoch_start",
        "on_fit_end",
        "on_configure_sharded_model",
        "on_fit_start",
        "on_init_end",
        "on_init_start",
        "on_keyboard_interrupt",
        "on_load_checkpoint",
        "on_pretrain_routine_end",
        "on_pretrain_routine_start",
        "on_sanity_check_end",
        "on_sanity_check_start",
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
        "on_predict_batch_end",
        "on_predict_batch_start",
        "on_predict_end",
        "on_predict_epoch_end",
        "on_predict_epoch_start",
        "on_predict_start",
        "on_save_checkpoint",
        "on_test_end",
        "on_train_end",
        "on_validation_end",
        "setup",
        "teardown",
    ]

    assert funcs_name == sorted(
        callbacks_func
    ), "Detected new callback function. Need to add its logging permission to FxValidator and update this test"

    validator = FxValidator()

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
            validator.check_logging(fx_name=func_name, on_step=on_step, on_epoch=on_epoch)
            if not is_start and is_stage:
                with pytest.raises(MisconfigurationException, match="You can't"):
                    validator.check_logging(fx_name=func_name, on_step=True, on_epoch=on_epoch)
        else:
            assert func_name in not_supported
            with pytest.raises(MisconfigurationException, match="function doesn't support"):
                validator.check_logging(fx_name=func_name, on_step=on_step, on_epoch=on_epoch)

    with pytest.raises(RuntimeError, match="`foo` but it is not implemented"):
        validator.check_logging("foo", False, False)


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
        default_root_dir=tmpdir, accelerator="dp", gpus=2, limit_train_batches=2, limit_val_batches=2, max_epochs=1
    )
    trainer.fit(model)
    trainer.test(model, ckpt_path=None)


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
    """Test that logging a metric with a reserved name to the progress bar raises a warning."""

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
    """test that auto_add_dataloader_idx argument works"""

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

    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=2)
    trainer.fit(model)
    logged = trainer.logged_metrics

    # Check that the correct keys exist
    if add_dataloader_idx:
        assert "val_loss/dataloader_idx_0" in logged
        assert "val_loss/dataloader_idx_1" in logged
    else:
        assert "val_loss_custom_naming_0" in logged
        assert "val_loss_custom_naming_1" in logged


def test_metrics_reset(tmpdir):
    """Tests that metrics are reset correctly after the end of the train/val/test epoch."""

    class TestModel(LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(32, 1)

        def _create_metrics(self):
            acc = Accuracy()
            acc.reset = mock.Mock(side_effect=acc.reset)
            ap = AveragePrecision(num_classes=1, pos_label=1)
            ap.reset = mock.Mock(side_effect=ap.reset)
            return acc, ap

        def setup(self, stage):
            fn = stage
            if fn == "fit":
                for stage in ("train", "validate"):
                    acc, ap = self._create_metrics()
                    self.add_module(f"acc_{fn}_{stage}", acc)
                    self.add_module(f"ap_{fn}_{stage}", ap)
            else:
                acc, ap = self._create_metrics()
                stage = self.trainer.state.stage
                self.add_module(f"acc_{fn}_{stage}", acc)
                self.add_module(f"ap_{fn}_{stage}", ap)

        def forward(self, x):
            return self.layer(x)

        def _step(self, batch):
            fn, stage = self.trainer.state.fn, self.trainer.state.stage

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
                return
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
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        max_epochs=1,
        progress_bar_refresh_rate=0,
        num_sanity_val_steps=2,
        checkpoint_callback=False,
    )

    trainer.fit(model)
    _assert_called(model, "fit", "train")
    _assert_called(model, "fit", "validate")

    trainer.validate(model)
    _assert_called(model, "validate", "validate")

    trainer.test(model)
    _assert_called(model, "test", "test")


def test_result_collection_on_tensor_with_mean_reduction():
    result_collection = ResultCollection(True, torch.device("cpu"))
    product = [(True, True), (False, True), (True, False), (False, False)]
    values = torch.arange(1, 10).float()  # need to convert to float() due to precision issues using torch 1.4
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
                    result_collection.log(
                        "training_step",
                        name,
                        v,
                        on_step=on_step,
                        on_epoch=on_epoch,
                        batch_size=batches[i],
                        prog_bar=prog_bar,
                        logger=logger,
                    )

    total_value = sum(values * batches)
    total_batches = sum(batches)
    assert result_collection["training_step.loss_on_step_on_epoch"].value == total_value
    assert result_collection["training_step.loss_on_step_on_epoch"].cumulated_batch_size == total_batches

    batch_metrics = result_collection.metrics(True)
    max_ = max(values)
    assert batch_metrics[MetricSource.PBAR] == {
        "loss_on_step_on_epoch_prog_bar_step": max_,
        "loss_on_step_on_epoch_prog_bar_logger_step": max_,
        "loss_on_step_prog_bar": max_,
        "loss_on_step_prog_bar_logger": max_,
    }
    assert batch_metrics[MetricSource.LOG] == {
        "loss_on_step_on_epoch_logger_step": max_,
        "loss_on_step_logger": max_,
        "loss_on_step_on_epoch_prog_bar_logger_step": max_,
        "loss_on_step_prog_bar_logger": max_,
    }
    assert batch_metrics[MetricSource.CALLBACK] == {
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
    assert epoch_metrics[MetricSource.PBAR] == {
        "loss_on_epoch_prog_bar": mean,
        "loss_on_epoch_prog_bar_logger": mean,
        "loss_on_step_on_epoch_prog_bar_epoch": mean,
        "loss_on_step_on_epoch_prog_bar_logger_epoch": mean,
    }
    assert epoch_metrics[MetricSource.LOG] == {
        "loss_on_epoch_logger": mean,
        "loss_on_epoch_prog_bar_logger": mean,
        "loss_on_step_on_epoch_logger_epoch": mean,
        "loss_on_step_on_epoch_prog_bar_logger_epoch": mean,
    }
    assert epoch_metrics[MetricSource.CALLBACK] == {
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


def test_logged_metrics_has_logged_epoch_value(tmpdir):
    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx):
            self.log("epoch", -batch_idx, logger=True)
            return super().training_step(batch, batch_idx)

    model = TestModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=2)
    trainer.fit(model)

    # should not get overridden if logged manually
    assert trainer.logged_metrics == {"epoch": -1}
