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
"""Test deprecated functionality which will be removed in v1.7.0."""
from unittest import mock

import pytest

from pytorch_lightning import Callback, LightningDataModule, Trainer
from pytorch_lightning.callbacks.gpu_stats_monitor import GPUStatsMonitor
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.progress import ProgressBar
from pytorch_lightning.callbacks.xla_stats_monitor import XLAStatsMonitor
from pytorch_lightning.loggers import LoggerCollection, TestTubeLogger
from tests.callbacks.test_callbacks import OldStatefulCallback
from tests.deprecated_api import _soft_unimport_module
from tests.helpers import BoringModel
from tests.helpers.datamodules import MNISTDataModule
from tests.helpers.runif import RunIf
from tests.loggers.test_base import CustomLogger


def test_v1_7_0_deprecated_lightning_module_summarize(tmpdir):
    from pytorch_lightning.core.lightning import warning_cache

    model = BoringModel()
    model.summarize(max_depth=1)
    assert any("The `LightningModule.summarize` method is deprecated in v1.5" in w for w in warning_cache)
    warning_cache.clear()


def test_v1_7_0_moved_model_summary_and_layer_summary(tmpdir):
    _soft_unimport_module("pytorch_lightning.core.memory")
    with pytest.deprecated_call(match="to `pytorch_lightning.utilities.model_summary` since v1.5"):
        from pytorch_lightning.core.memory import LayerSummary, ModelSummary  # noqa: F401


def test_v1_7_0_moved_get_memory_profile_and_get_gpu_memory_map(tmpdir):
    _soft_unimport_module("pytorch_lightning.core.memory")
    with pytest.deprecated_call(match="to `pytorch_lightning.utilities.memory` since v1.5"):
        from pytorch_lightning.core.memory import get_gpu_memory_map, get_memory_profile  # noqa: F401


def test_v1_7_0_deprecated_model_size():
    model = BoringModel()
    with pytest.deprecated_call(
        match="LightningModule.model_size` property was deprecated in v1.5 and will be removed in v1.7"
    ):
        _ = model.model_size


def test_v1_7_0_datamodule_transform_properties(tmpdir):
    dm = MNISTDataModule()
    with pytest.deprecated_call(match=r"DataModule property `train_transforms` was deprecated in v1.5"):
        dm.train_transforms = "a"
    with pytest.deprecated_call(match=r"DataModule property `val_transforms` was deprecated in v1.5"):
        dm.val_transforms = "b"
    with pytest.deprecated_call(match=r"DataModule property `test_transforms` was deprecated in v1.5"):
        dm.test_transforms = "c"
    with pytest.deprecated_call(match=r"DataModule property `train_transforms` was deprecated in v1.5"):
        _ = LightningDataModule(train_transforms="a")
    with pytest.deprecated_call(match=r"DataModule property `val_transforms` was deprecated in v1.5"):
        _ = LightningDataModule(val_transforms="b")
    with pytest.deprecated_call(match=r"DataModule property `test_transforms` was deprecated in v1.5"):
        _ = LightningDataModule(test_transforms="c")
    with pytest.deprecated_call(match=r"DataModule property `test_transforms` was deprecated in v1.5"):
        _ = LightningDataModule(test_transforms="c", dims=(1, 1, 1))


def test_v1_7_0_datamodule_size_property(tmpdir):
    dm = MNISTDataModule()
    with pytest.deprecated_call(match=r"DataModule property `size` was deprecated in v1.5"):
        dm.size()


def test_v1_7_0_datamodule_dims_property(tmpdir):
    dm = MNISTDataModule()
    with pytest.deprecated_call(match=r"DataModule property `dims` was deprecated in v1.5"):
        _ = dm.dims
    with pytest.deprecated_call(match=r"DataModule property `dims` was deprecated in v1.5"):
        _ = LightningDataModule(dims=(1, 1, 1))


def test_v1_7_0_moved_get_progress_bar_dict(tmpdir):
    class TestModel(BoringModel):
        def get_progress_bar_dict(self):
            items = super().get_progress_bar_dict()
            items.pop("v_num", None)
            return items

    trainer = Trainer(
        default_root_dir=tmpdir,
        progress_bar_refresh_rate=None,
        fast_dev_run=True,
    )
    test_model = TestModel()
    with pytest.deprecated_call(match=r"`LightningModule.get_progress_bar_dict` method was deprecated in v1.5"):
        trainer.fit(test_model)
    standard_metrics_postfix = trainer.progress_bar_callback.main_progress_bar.postfix
    assert "loss" in standard_metrics_postfix
    assert "v_num" not in standard_metrics_postfix

    with pytest.deprecated_call(match=r"`trainer.progress_bar_dict` is deprecated in v1.5"):
        _ = trainer.progress_bar_dict


def test_v1_7_0_trainer_prepare_data_per_node(tmpdir):
    with pytest.deprecated_call(
        match="Setting `prepare_data_per_node` with the trainer flag is deprecated and will be removed in v1.7.0!"
    ):
        _ = Trainer(prepare_data_per_node=False)


def test_v1_7_0_stochastic_weight_avg_trainer_constructor(tmpdir):
    with pytest.deprecated_call(match=r"Setting `Trainer\(stochastic_weight_avg=True\)` is deprecated in v1.5"):
        _ = Trainer(stochastic_weight_avg=True)


@pytest.mark.parametrize("terminate_on_nan", [True, False])
def test_v1_7_0_trainer_terminate_on_nan(tmpdir, terminate_on_nan):
    with pytest.deprecated_call(
        match="Trainer argument `terminate_on_nan` was deprecated in v1.5 and will be removed in 1.7"
    ):
        trainer = Trainer(terminate_on_nan=terminate_on_nan)
        assert trainer.terminate_on_nan is terminate_on_nan
        assert trainer._detect_anomaly is False

    trainer = Trainer()
    with pytest.deprecated_call(match=r"`Trainer.terminate_on_nan` is deprecated in v1.5"):
        _ = trainer.terminate_on_nan

    with pytest.deprecated_call(match=r"Setting `Trainer.terminate_on_nan = True` is deprecated in v1.5"):
        trainer.terminate_on_nan = True


def test_v1_7_0_deprecated_on_task_dataloader(tmpdir):
    class CustomBoringModel(BoringModel):
        def on_train_dataloader(self):
            print("on_train_dataloader")

        def on_val_dataloader(self):
            print("on_val_dataloader")

        def on_test_dataloader(self):
            print("on_test_dataloader")

        def on_predict_dataloader(self):
            print("on_predict_dataloader")

    def _run(model, task="fit"):
        trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=2)
        getattr(trainer, task)(model)

    model = CustomBoringModel()

    with pytest.deprecated_call(
        match="Method `on_train_dataloader` is deprecated in v1.5.0 and will be removed in v1.7.0."
    ):
        _run(model, "fit")

    with pytest.deprecated_call(
        match="Method `on_val_dataloader` is deprecated in v1.5.0 and will be removed in v1.7.0."
    ):
        _run(model, "fit")

    with pytest.deprecated_call(
        match="Method `on_val_dataloader` is deprecated in v1.5.0 and will be removed in v1.7.0."
    ):
        _run(model, "validate")

    with pytest.deprecated_call(
        match="Method `on_test_dataloader` is deprecated in v1.5.0 and will be removed in v1.7.0."
    ):
        _run(model, "test")

    with pytest.deprecated_call(
        match="Method `on_predict_dataloader` is deprecated in v1.5.0 and will be removed in v1.7.0."
    ):
        _run(model, "predict")


@mock.patch("pytorch_lightning.loggers.test_tube.Experiment")
def test_v1_7_0_test_tube_logger(_, tmpdir):
    with pytest.deprecated_call(match="The TestTubeLogger is deprecated since v1.5 and will be removed in v1.7"):
        _ = TestTubeLogger(tmpdir)


def test_v1_7_0_on_interrupt(tmpdir):
    class HandleInterruptCallback(Callback):
        def on_keyboard_interrupt(self, trainer, pl_module):
            print("keyboard interrupt")

    model = BoringModel()
    handle_interrupt_callback = HandleInterruptCallback()

    trainer = Trainer(
        callbacks=[handle_interrupt_callback],
        max_epochs=1,
        limit_val_batches=0.1,
        limit_train_batches=0.2,
        enable_progress_bar=False,
        logger=False,
        default_root_dir=tmpdir,
    )
    with pytest.deprecated_call(
        match="The `on_keyboard_interrupt` callback hook was deprecated in v1.5 and will be removed in v1.7"
    ):
        trainer.fit(model)


def test_v1_7_0_process_position_trainer_constructor(tmpdir):
    with pytest.deprecated_call(match=r"Setting `Trainer\(process_position=5\)` is deprecated in v1.5"):
        _ = Trainer(process_position=5)


def test_v1_7_0_flush_logs_every_n_steps_trainer_constructor(tmpdir):
    with pytest.deprecated_call(match=r"Setting `Trainer\(flush_logs_every_n_steps=10\)` is deprecated in v1.5"):
        _ = Trainer(flush_logs_every_n_steps=10)


class BoringCallbackDDPSpawnModel(BoringModel):
    def add_to_queue(self, queue):
        ...

    def get_from_queue(self, queue):
        ...


def test_v1_7_0_deprecate_add_get_queue(tmpdir):
    model = BoringCallbackDDPSpawnModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)

    with pytest.deprecated_call(match=r"`LightningModule.add_to_queue` method was deprecated in v1.5"):
        trainer.fit(model)

    with pytest.deprecated_call(match=r"`LightningModule.get_from_queue` method was deprecated in v1.5"):
        trainer.fit(model)


def test_v1_7_0_progress_bar_refresh_rate_trainer_constructor(tmpdir):
    with pytest.deprecated_call(match=r"Setting `Trainer\(progress_bar_refresh_rate=1\)` is deprecated in v1.5"):
        _ = Trainer(progress_bar_refresh_rate=1)


def test_v1_7_0_lightning_logger_base_close(tmpdir):
    logger = CustomLogger()
    with pytest.deprecated_call(
        match="`LightningLoggerBase.close` method is deprecated in v1.5 and will be removed in v1.7."
    ):
        logger.close()
    with pytest.deprecated_call(
        match="`LoggerCollection.close` method is deprecated in v1.5 and will be removed in v1.7."
    ):
        logger = LoggerCollection([logger])
        logger.close()


def test_v1_7_0_deprecate_lightning_distributed(tmpdir):
    with pytest.deprecated_call(match="LightningDistributed is deprecated in v1.5 and will be removed in v1.7."):
        from pytorch_lightning.distributed.dist import LightningDistributed

        _ = LightningDistributed()


def test_v1_7_0_checkpoint_callback_trainer_constructor(tmpdir):
    with pytest.deprecated_call(match=r"Setting `Trainer\(checkpoint_callback=True\)` is deprecated in v1.5"):
        _ = Trainer(checkpoint_callback=True)


def test_v1_7_0_old_on_train_batch_start(tmpdir):
    class OldSignature(Callback):
        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
            ...

    class OldSignatureModel(BoringModel):
        def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
            ...

    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, callbacks=OldSignature())
    with pytest.deprecated_call(match="`dataloader_idx` argument will be removed in v1.7."):
        trainer.fit(model)

    model = OldSignatureModel()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1)
    with pytest.deprecated_call(match="`dataloader_idx` argument will be removed in v1.7."):
        trainer.fit(model)


def test_v1_7_0_old_on_train_batch_end(tmpdir):
    class OldSignature(Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
            ...

    class OldSignatureModel(BoringModel):
        def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
            ...

    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, callbacks=OldSignature(), fast_dev_run=True)
    with pytest.deprecated_call(match="`dataloader_idx` argument will be removed in v1.7."):
        trainer.fit(model)

    model = OldSignatureModel()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, callbacks=OldSignature(), fast_dev_run=True)
    with pytest.deprecated_call(match="`dataloader_idx` argument will be removed in v1.7."):
        trainer.fit(model)


def test_v1_7_0_deprecate_on_post_move_to_device(tmpdir):
    class TestModel(BoringModel):
        def on_post_move_to_device(self):
            print("on_post_move_to_device")

    model = TestModel()

    trainer = Trainer(default_root_dir=tmpdir, limit_train_batches=5, max_epochs=1)

    with pytest.deprecated_call(
        match=r"Method `on_post_move_to_device` has been deprecated in v1.5 and will be removed in v1.7"
    ):
        trainer.fit(model)


def test_v1_7_0_deprecate_parameter_validation():

    _soft_unimport_module("pytorch_lightning.core.decorators")
    with pytest.deprecated_call(
        match="Using `pytorch_lightning.core.decorators.parameter_validation` is deprecated in v1.5"
    ):
        from pytorch_lightning.core.decorators import parameter_validation  # noqa: F401


def test_v1_7_0_passing_strategy_to_accelerator_trainer_flag():
    with pytest.deprecated_call(match="has been deprecated in v1.5 and will be removed in v1.7."):
        Trainer(accelerator="ddp_spawn")


def test_v1_7_0_passing_strategy_to_plugins_flag():
    with pytest.deprecated_call(match="has been deprecated in v1.5 and will be removed in v1.7."):
        Trainer(plugins="ddp_spawn")


def test_v1_7_0_weights_summary_trainer(tmpdir):
    with pytest.deprecated_call(match=r"Setting `Trainer\(weights_summary=full\)` is deprecated in v1.5"):
        t = Trainer(weights_summary="full")

    with pytest.deprecated_call(match=r"Setting `Trainer\(weights_summary=None\)` is deprecated in v1.5"):
        t = Trainer(weights_summary=None)

    t = Trainer(weights_summary="top")
    with pytest.deprecated_call(match=r"`Trainer.weights_summary` is deprecated in v1.5"):
        _ = t.weights_summary

    with pytest.deprecated_call(match=r"Setting `Trainer.weights_summary` is deprecated in v1.5"):
        t.weights_summary = "blah"


def test_v1_7_0_trainer_log_gpu_memory(tmpdir):
    with pytest.deprecated_call(
        match="Setting `log_gpu_memory` with the trainer flag is deprecated in v1.5 and will be removed"
    ):
        _ = Trainer(log_gpu_memory="min_max")


@RunIf(min_gpus=1)
def test_v1_7_0_deprecate_gpu_stats_monitor(tmpdir):
    with pytest.deprecated_call(match="The `GPUStatsMonitor` callback was deprecated in v1.5"):
        _ = GPUStatsMonitor()


@RunIf(tpu=True)
def test_v1_7_0_deprecate_xla_stats_monitor(tmpdir):
    with pytest.deprecated_call(match="The `XLAStatsMonitor` callback was deprecated in v1.5"):
        _ = XLAStatsMonitor()


def test_v1_7_0_progress_bar():
    with pytest.deprecated_call(match="has been deprecated in v1.5 and will be removed in v1.7."):
        _ = ProgressBar()


def test_v1_7_0_deprecated_max_steps_none(tmpdir):
    with pytest.deprecated_call(match="`max_steps = None` is deprecated in v1.5"):
        _ = Trainer(max_steps=None)

    trainer = Trainer()
    with pytest.deprecated_call(match="`max_steps = None` is deprecated in v1.5"):
        trainer.fit_loop.max_steps = None


def test_v1_7_0_resume_from_checkpoint_trainer_constructor(tmpdir):
    with pytest.deprecated_call(match=r"Setting `Trainer\(resume_from_checkpoint=\)` is deprecated in v1.5"):
        trainer = Trainer(resume_from_checkpoint="a")
    with pytest.deprecated_call(
        match=r"trainer.resume_from_checkpoint` is deprecated in v1.5 and will be removed in v1.7."
    ):
        _ = trainer.resume_from_checkpoint

    # test resume_from_checkpoint still works until v1.7 deprecation
    model = BoringModel()
    callback = OldStatefulCallback(state=111)
    trainer = Trainer(default_root_dir=tmpdir, max_steps=1, callbacks=[callback])
    trainer.fit(model)
    ckpt_path = trainer.checkpoint_callback.best_model_path

    callback = OldStatefulCallback(state=222)
    trainer = Trainer(default_root_dir=tmpdir, max_steps=2, callbacks=[callback], resume_from_checkpoint=ckpt_path)
    assert trainer.checkpoint_connector.resume_checkpoint_path is None
    assert trainer.checkpoint_connector.resume_from_checkpoint_fit_path == ckpt_path
    trainer.validate(model=model, ckpt_path=ckpt_path)
    assert callback.state == 222
    assert trainer.checkpoint_connector.resume_checkpoint_path is None
    assert trainer.checkpoint_connector.resume_from_checkpoint_fit_path == ckpt_path
    trainer.fit(model)
    assert callback.state == 111
    assert trainer.checkpoint_connector.resume_checkpoint_path is None
    assert trainer.checkpoint_connector.resume_from_checkpoint_fit_path is None
    trainer.predict(model=model, ckpt_path=ckpt_path)
    assert trainer.checkpoint_connector.resume_checkpoint_path is None
    assert trainer.checkpoint_connector.resume_from_checkpoint_fit_path is None
    trainer.fit(model)
    assert trainer.checkpoint_connector.resume_checkpoint_path is None
    assert trainer.checkpoint_connector.resume_from_checkpoint_fit_path is None

    # test fit(ckpt_path=) precedence over Trainer(resume_from_checkpoint=) path
    model = BoringModel()
    trainer = Trainer(resume_from_checkpoint="trainer_arg_path")
    with pytest.raises(FileNotFoundError, match="Checkpoint at fit_arg_ckpt_path not found. Aborting training."):
        trainer.fit(model, ckpt_path="fit_arg_ckpt_path")


def test_v1_7_0_deprecate_lr_sch_names(tmpdir):
    model = BoringModel()
    lr_monitor = LearningRateMonitor()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True, callbacks=[lr_monitor])
    trainer.fit(model)

    with pytest.deprecated_call(match="`LearningRateMonitor.lr_sch_names` has been deprecated in v1.5"):
        assert lr_monitor.lr_sch_names == ["lr-SGD"]
