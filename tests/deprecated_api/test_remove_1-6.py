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
"""Test deprecated functionality which will be removed in v1.6.0."""
from unittest.mock import call, Mock

import pytest
import torch
from torch.optim import Optimizer

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.plugins import PrecisionPlugin
from pytorch_lightning.plugins.training_type import DDPPlugin
from pytorch_lightning.utilities.distributed import rank_zero_deprecation, rank_zero_warn
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.model_summary import ModelSummary
from tests.deprecated_api import _soft_unimport_module
from tests.helpers import BoringModel


def test_old_transfer_batch_to_device_hook(tmpdir):
    class OldModel(BoringModel):
        def transfer_batch_to_device(self, batch, device):
            return super().transfer_batch_to_device(batch, device, None)

    trainer = Trainer(default_root_dir=tmpdir, limit_train_batches=1, limit_val_batches=0, max_epochs=1)
    with pytest.deprecated_call(match="old signature will be removed in v1.6"):
        trainer.fit(OldModel())


def test_v1_6_0_reload_dataloaders_every_epoch(tmpdir):
    model = BoringModel()

    tracker = Mock()
    model.train_dataloader = Mock(wraps=model.train_dataloader)
    model.val_dataloader = Mock(wraps=model.val_dataloader)
    model.test_dataloader = Mock(wraps=model.test_dataloader)

    tracker.attach_mock(model.train_dataloader, "train_dataloader")
    tracker.attach_mock(model.val_dataloader, "val_dataloader")
    tracker.attach_mock(model.test_dataloader, "test_dataloader")

    with pytest.deprecated_call(match="`reload_dataloaders_every_epoch` is deprecated in v1.4 and will be removed"):
        trainer = Trainer(
            default_root_dir=tmpdir,
            limit_train_batches=0.3,
            limit_val_batches=0.3,
            reload_dataloaders_every_epoch=True,
            max_epochs=3,
        )
    trainer.fit(model)
    trainer.test()

    expected_sequence = (
        [call.val_dataloader()] + [call.train_dataloader(), call.val_dataloader()] * 3 + [call.test_dataloader()]
    )
    assert tracker.mock_calls == expected_sequence


def test_v1_6_0_tbptt_reduce_fx(tmpdir):
    class TestModel(BoringModel):
        def training_step(self, *args):
            self.log("foo", 1, tbptt_reduce_fx=lambda x: x)
            return super().training_step(*args)

    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    with pytest.deprecated_call(match=r"tbptt_reduce_fx=...\)` is no longer supported"):
        trainer.fit(TestModel())


def test_v1_6_0_tbptt_pad_token(tmpdir):
    class TestModel(BoringModel):
        def training_step(self, *args):
            self.log("foo", 1, tbptt_pad_token=0)
            return super().training_step(*args)

    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    with pytest.deprecated_call(match=r"tbptt_pad_token=...\)` is no longer supported"):
        trainer.fit(TestModel())


def test_v1_6_0_sync_dist_op(tmpdir):
    class TestModel(BoringModel):
        def training_step(self, *args):
            self.log("foo", 1, sync_dist_op="sum")
            return super().training_step(*args)

    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    with pytest.deprecated_call(match=r"`self.log\(sync_dist_op='sum'\)` is deprecated"):
        trainer.fit(TestModel())


def test_v1_6_0_is_overridden_model():
    model = BoringModel()
    with pytest.deprecated_call(match="and will be removed in v1.6"):
        assert is_overridden("validation_step", model=model)
    with pytest.deprecated_call(match="and will be removed in v1.6"):
        assert not is_overridden("foo", model=model)


def test_v1_6_0_early_stopping_monitor(tmpdir):
    with pytest.deprecated_call(
        match=r"The `EarlyStopping\(monitor\)` argument will be required starting in v1.6."
        " For backward compatibility, setting this to `early_stop_on`."
    ):
        EarlyStopping()


def test_v1_6_0_extras_with_gradients(tmpdir):
    class TestModel(BoringModel):
        def training_step(self, *args):
            loss = super().training_step(*args)["loss"]
            return {"loss": loss, "foo": loss}

    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=1)
    model = TestModel()
    match = r"\{'foo'\} has a `grad_fn`.*behaviour will change in v1\.6"
    with pytest.deprecated_call(match=match):
        trainer.fit(model)


def test_v1_6_0_train_loop(tmpdir):
    trainer = Trainer()
    with pytest.deprecated_call(
        match=r"`Trainer.train_loop` has been renamed to `Trainer.fit_loop` and will be removed in v1.6."
    ):
        _ = trainer.train_loop


def test_v1_6_0_rank_zero_warnings_moved():
    with pytest.deprecated_call(match="in v1.3.7 and will be removed in v1.6"):
        rank_zero_warn("test")
    with pytest.deprecated_call(match="in v1.3.7 and will be removed in v1.6"):
        rank_zero_deprecation("test")


def test_v1_6_0_ddp_plugin_task_idx():
    plugin = DDPPlugin()
    with pytest.deprecated_call(match="Use `DDPPlugin.local_rank` instead"):
        _ = plugin.task_idx


def test_v1_6_0_deprecated_model_summary_mode(tmpdir):
    model = BoringModel()
    with pytest.deprecated_call(match="Argument `mode` in `ModelSummary` is deprecated in v1.4"):
        ModelSummary(model, mode="top")

    with pytest.deprecated_call(match="Argument `mode` in `LightningModule.summarize` is deprecated in v1.4"):
        model.summarize(mode="top")


def test_v1_6_0_deprecated_disable_validation():
    trainer = Trainer()
    with pytest.deprecated_call(match="disable_validation` is deprecated in v1.4"):
        _ = trainer.disable_validation


def test_v1_6_0_every_n_val_epochs():
    with pytest.deprecated_call(match="use `every_n_epochs` instead"):
        _ = ModelCheckpoint(every_n_val_epochs=1)


def test_v1_6_0_deprecated_hpc_load(tmpdir):
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, max_steps=1)
    trainer.fit(model)
    trainer.checkpoint_connector.hpc_save(tmpdir, trainer.logger)
    checkpoint_path = trainer.checkpoint_connector.get_max_ckpt_path_from_folder(str(tmpdir))
    with pytest.deprecated_call(match=r"`CheckpointConnector.hpc_load\(\)` was deprecated in v1.4"):
        trainer.checkpoint_connector.hpc_load(checkpoint_path)


def test_v1_6_0_deprecated_device_dtype_mixin_import():

    _soft_unimport_module("pytorch_lightning.utilities.device_dtype_mixin")
    with pytest.deprecated_call(match="will be removed in v1.6"):
        from pytorch_lightning.utilities.device_dtype_mixin import DeviceDtypeModuleMixin  # noqa: F401


def test_v1_6_0_deprecated_accelerator_pass_through_functions():
    from pytorch_lightning.plugins.precision import PrecisionPlugin
    from pytorch_lightning.plugins.training_type import SingleDevicePlugin

    plugin = SingleDevicePlugin(torch.device("cpu"))
    from pytorch_lightning.accelerators.accelerator import Accelerator

    accelerator = Accelerator(training_type_plugin=plugin, precision_plugin=PrecisionPlugin())
    with pytest.deprecated_call(match="will be removed in v1.6"):
        accelerator.barrier()

    with pytest.deprecated_call(match="will be removed in v1.6"):
        accelerator.broadcast(1)

    with pytest.deprecated_call(match="will be removed in v1.6"):
        tensor = torch.rand(2, 2, requires_grad=True)
        accelerator.all_gather(tensor)

    with pytest.deprecated_call(match="will be removed in v1.6"):
        model = BoringModel()
        accelerator.connect(model)

    with pytest.deprecated_call(match="will be removed in v1.6"):
        accelerator.post_training_step()

    with pytest.deprecated_call(match="will be removed in v1.6"):
        tensor = torch.rand(2, 2, requires_grad=True)
        accelerator.training_step_end(tensor)

    with pytest.deprecated_call(match="will be removed in v1.6"):
        tensor = torch.rand(2, 2, requires_grad=True)
        accelerator.test_step_end(tensor)

    with pytest.deprecated_call(match="will be removed in v1.6"):
        tensor = torch.rand(2, 2, requires_grad=True)
        accelerator.validation_step_end(tensor)

    with pytest.deprecated_call(match="will be removed in v1.6"):
        accelerator.lightning_module_state_dict()

    with pytest.deprecated_call(match="will be removed in v1.6"):
        dl = model.train_dataloader()
        accelerator.process_dataloader(dl)

    with pytest.deprecated_call(match="will be removed in v1.6"):
        accelerator.results

    with pytest.deprecated_call(match="will be removed in v1.6"):
        accelerator.setup_optimizers_in_pre_dispatch

    with pytest.deprecated_call(match="will be removed in v1.6"):
        accelerator.restore_checkpoint_after_pre_dispatch

    with pytest.deprecated_call(match="will be removed in v1.6"):
        accelerator.on_validation_start()

    with pytest.deprecated_call(match="will be removed in v1.6"):
        accelerator.on_test_start()

    with pytest.deprecated_call(match="will be removed in v1.6"):
        accelerator.on_predict_start()

    with pytest.deprecated_call(match="will be removed in v1.6"):
        accelerator.on_validation_end()

    with pytest.deprecated_call(match="will be removed in v1.6"):
        accelerator.on_test_end()

    with pytest.deprecated_call(match="will be removed in v1.6"):
        accelerator.on_predict_end()

    with pytest.deprecated_call(match="will be removed in v1.6"):
        accelerator.on_train_end()

    with pytest.deprecated_call(match="will be removed in v1.6"):
        accelerator.on_train_batch_start(batch=None, batch_idx=0)
