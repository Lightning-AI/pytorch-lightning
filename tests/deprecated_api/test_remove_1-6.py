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
""" Test deprecated functionality which will be removed in v1.6.0 """
import pytest

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.core.memory import ModelSummary
from pytorch_lightning.plugins.training_type import DDPPlugin, DDPSpawnPlugin
from pytorch_lightning.utilities.distributed import rank_zero_deprecation, rank_zero_warn
from pytorch_lightning.utilities.model_helpers import is_overridden
from tests.deprecated_api import _soft_unimport_module
from tests.helpers import BoringDataModule, BoringModel


def test_v1_6_0_trainer_model_hook_mixin(tmpdir):
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, checkpoint_callback=False, logger=False)
    trainer.fit(model)
    with pytest.deprecated_call(match="is deprecated in v1.4 and will be removed in v1.6"):
        trainer.is_function_implemented("training_step", model)

    with pytest.deprecated_call(match="is deprecated in v1.4 and will be removed in v1.6"):
        trainer.has_arg("training_step", "batch")


def test_v1_6_0_dataloader_renaming(tmpdir):
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    dl = model.train_dataloader()

    with pytest.deprecated_call(match=r"fit\(train_dataloader\)` is deprecated in v1.4"):
        trainer.fit(model, train_dataloader=dl)

    with pytest.deprecated_call(match=r"validate\(val_dataloaders\)` is deprecated in v1.4"):
        trainer.validate(model, val_dataloaders=dl)

    with pytest.deprecated_call(match=r"test\(test_dataloaders\)` is deprecated in v1.4"):
        trainer.test(model, test_dataloaders=dl)

    with pytest.deprecated_call(match=r"tune\(train_dataloader\)` is deprecated in v1.4"):
        trainer.tune(model, train_dataloader=dl)
    with pytest.deprecated_call(match=r"tune\(train_dataloader\)` is deprecated in v1.4"):
        trainer.tuner.scale_batch_size(model, train_dataloader=dl)
    with pytest.deprecated_call(match=r"tune\(train_dataloader\)` is deprecated in v1.4"):
        trainer.tuner.lr_find(model, train_dataloader=dl)


def test_old_transfer_batch_to_device_hook(tmpdir):
    class OldModel(BoringModel):
        def transfer_batch_to_device(self, batch, device):
            return super().transfer_batch_to_device(batch, device, None)

    trainer = Trainer(default_root_dir=tmpdir, limit_train_batches=1, limit_val_batches=0, max_epochs=1)
    with pytest.deprecated_call(match="old signature will be removed in v1.6"):
        trainer.fit(OldModel())


def test_v1_6_0_ddp_num_nodes():
    with pytest.deprecated_call(match="Argument `num_nodes` in `DDPPlugin` is deprecated in v1.4"):
        DDPPlugin(num_nodes=1)


def test_v1_6_0_ddp_sync_batchnorm():
    with pytest.deprecated_call(match="Argument `sync_batchnorm` in `DDPPlugin` is deprecated in v1.4"):
        DDPPlugin(sync_batchnorm=False)


def test_v1_6_0_ddp_spawn_num_nodes():
    with pytest.deprecated_call(match="Argument `num_nodes` in `DDPSpawnPlugin` is deprecated in v1.4"):
        DDPSpawnPlugin(num_nodes=1)


def test_v1_6_0_ddp_spawn_sync_batchnorm():
    with pytest.deprecated_call(match="Argument `sync_batchnorm` in `DDPSpawnPlugin` is deprecated in v1.4"):
        DDPSpawnPlugin(sync_batchnorm=False)


def test_v1_6_0_reload_dataloaders_every_epoch(tmpdir):

    model = BoringModel()

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

    # verify the sequence
    calls = trainer.dev_debugger.dataloader_sequence_calls
    expected_sequence = ["val_dataloader"] + ["train_dataloader", "val_dataloader"] * 3 + ["test_dataloader"]
    for call, expected in zip(calls, expected_sequence):
        assert call["name"] == expected


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


def test_v1_6_0_datamodule_lifecycle_properties(tmpdir):
    dm = BoringDataModule()
    with pytest.deprecated_call(match=r"DataModule property `has_prepared_data` was deprecated in v1.4"):
        dm.has_prepared_data
    with pytest.deprecated_call(match=r"DataModule property `has_setup_fit` was deprecated in v1.4"):
        dm.has_setup_fit
    with pytest.deprecated_call(match=r"DataModule property `has_setup_validate` was deprecated in v1.4"):
        dm.has_setup_validate
    with pytest.deprecated_call(match=r"DataModule property `has_setup_test` was deprecated in v1.4"):
        dm.has_setup_test
    with pytest.deprecated_call(match=r"DataModule property `has_setup_predict` was deprecated in v1.4"):
        dm.has_setup_predict
    with pytest.deprecated_call(match=r"DataModule property `has_teardown_fit` was deprecated in v1.4"):
        dm.has_teardown_fit
    with pytest.deprecated_call(match=r"DataModule property `has_teardown_validate` was deprecated in v1.4"):
        dm.has_teardown_validate
    with pytest.deprecated_call(match=r"DataModule property `has_teardown_test` was deprecated in v1.4"):
        dm.has_teardown_test
    with pytest.deprecated_call(match=r"DataModule property `has_teardown_predict` was deprecated in v1.4"):
        dm.has_teardown_predict


def test_v1_6_0_datamodule_hooks_calls(tmpdir):
    """Test that repeated calls to DataHooks' hooks show a warning about the coming API change."""

    class TestDataModule(BoringDataModule):
        setup_calls = []
        teardown_calls = []
        prepare_data_calls = 0

        def setup(self, stage=None):
            super().setup(stage=stage)
            self.setup_calls.append(stage)

        def teardown(self, stage=None):
            super().teardown(stage=stage)
            self.teardown_calls.append(stage)

        def prepare_data(self):
            super().prepare_data()
            self.prepare_data_calls += 1

    dm = TestDataModule()
    dm.prepare_data()
    dm.prepare_data()
    dm.setup("fit")
    with pytest.deprecated_call(
        match=r"DataModule.setup has already been called, so it will not be called again. "
        "In v1.6 this behavior will change to always call DataModule.setup"
    ):
        dm.setup("fit")
    dm.setup()
    dm.setup()
    dm.teardown("validate")
    with pytest.deprecated_call(
        match=r"DataModule.teardown has already been called, so it will not be called again. "
        "In v1.6 this behavior will change to always call DataModule.teardown"
    ):
        dm.teardown("validate")

    assert dm.prepare_data_calls == 1
    assert dm.setup_calls == ["fit", None]
    assert dm.teardown_calls == ["validate"]

    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=1)
    trainer.test(BoringModel(), datamodule=dm)

    # same number of calls
    assert dm.prepare_data_calls == 1
    assert dm.setup_calls == ["fit", None]
    assert dm.teardown_calls == ["validate", "test"]


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


def test_v1_6_0_lightning_module_loaded_optimizer_states_dict():
    from pytorch_lightning.core.lightning import warning_cache

    model = BoringModel()
    _ = model.loaded_optimizer_states_dict
    assert any(
        "The `LightningModule.loaded_optimizer_states_dict` property is deprecated in v1.4" in w for w in warning_cache
    )
    warning_cache.clear()

    model.loaded_optimizer_states_dict = {}
    assert any(
        "The `LightningModule.loaded_optimizer_states_dict` property is deprecated in v1.4" in w for w in warning_cache
    )
    warning_cache.clear()


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
        from pytorch_lightning.utilities.device_dtype_mixin import DeviceDtypeModuleMixin  # noqa: F811 F401
