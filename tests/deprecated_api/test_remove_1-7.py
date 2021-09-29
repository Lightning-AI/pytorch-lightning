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
import torch
from torch import nn

from pytorch_lightning import Callback, LightningDataModule, Trainer
from pytorch_lightning.loggers import LoggerCollection, TestTubeLogger
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
        match="Method `on_train_dataloader` in DataHooks is deprecated and will be removed in v1.7.0."
    ):
        _run(model, "fit")

    with pytest.deprecated_call(
        match="Method `on_val_dataloader` in DataHooks is deprecated and will be removed in v1.7.0."
    ):
        _run(model, "fit")

    with pytest.deprecated_call(
        match="Method `on_val_dataloader` in DataHooks is deprecated and will be removed in v1.7.0."
    ):
        _run(model, "validate")

    with pytest.deprecated_call(
        match="Method `on_test_dataloader` in DataHooks is deprecated and will be removed in v1.7.0."
    ):
        _run(model, "test")

    with pytest.deprecated_call(
        match="Method `on_predict_dataloader` in DataHooks is deprecated and will be removed in v1.7.0."
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
    def __init__(self):
        super().__init__()

    def add_to_queue(self, queue: torch.multiprocessing.SimpleQueue) -> None:
        queue.put("test_val")
        return super().add_to_queue(queue)

    def get_from_queue(self, queue: torch.multiprocessing.SimpleQueue) -> None:
        self.test_val = queue.get()
        return super().get_from_queue(queue)


@RunIf(skip_windows=True)
def test_v1_7_0_deprecate_add_get_queue(tmpdir):
    model = BoringCallbackDDPSpawnModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True, num_processes=2, accelerator="ddp_cpu")

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


def test_v1_7_0_deprecate_on_post_move_to_device(tmpdir):
    class WeightSharingModule(BoringModel):
        def __init__(self):
            super().__init__()
            self.layer_1 = nn.Linear(32, 10, bias=False)
            self.layer_2 = nn.Linear(10, 32, bias=False)
            self.layer_3 = nn.Linear(32, 10, bias=False)

        def on_post_move_to_device(self):
            self.layer_3.weight = self.layer_1.weight

        def forward(self, x):
            x = self.layer_1(x)
            x = self.layer_2(x)
            x = self.layer_3(x)
            return x

    model = WeightSharingModule()
    trainer = Trainer(default_root_dir=tmpdir, limit_train_batches=5, max_epochs=1)

    with pytest.deprecated_call(
        match=r"Method `on_post_move_to_device` has been deprecated in v1.5 and will be removed in v1.7"
    ):
        trainer.fit(model)
