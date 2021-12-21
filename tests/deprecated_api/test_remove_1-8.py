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
"""Test deprecated functionality which will be removed in v1.8.0."""
from unittest.mock import Mock

import pytest
import torch
from torch import optim

from pytorch_lightning import Callback, Trainer
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.apply_func import move_data_to_device
from pytorch_lightning.utilities.enums import DeviceType, DistributedType
from pytorch_lightning.utilities.imports import _TORCHTEXT_LEGACY
from tests.helpers.boring_model import BoringModel
from tests.helpers.torchtext_utils import get_dummy_torchtext_data_iterator


def test_v1_8_0_deprecated_distributed_type_enum():

    with pytest.deprecated_call(match="has been deprecated in v1.6 and will be removed in v1.8."):
        _ = DistributedType.DDP


def test_v1_8_0_deprecated_device_type_enum():

    with pytest.deprecated_call(match="has been deprecated in v1.6 and will be removed in v1.8."):
        _ = DeviceType.CPU


@pytest.mark.skipif(not _TORCHTEXT_LEGACY, reason="torchtext.legacy is deprecated.")
def test_v1_8_0_deprecated_torchtext_batch():

    with pytest.deprecated_call(match="is deprecated and Lightning will remove support for it in v1.8"):
        data_iterator, _ = get_dummy_torchtext_data_iterator(num_samples=3, batch_size=3)
        batch = next(iter(data_iterator))
        _ = move_data_to_device(batch=batch, device=torch.device("cpu"))


def test_v1_8_0_on_init_start_end(tmpdir):
    class TestCallback(Callback):
        def on_init_start(self, trainer):
            print("Starting to init trainer!")

        def on_init_end(self, trainer):
            print("Trainer is init now")

    model = BoringModel()

    trainer = Trainer(
        callbacks=[TestCallback()],
        max_epochs=1,
        fast_dev_run=True,
        enable_progress_bar=False,
        logger=False,
        default_root_dir=tmpdir,
    )
    with pytest.deprecated_call(
        match="The `on_init_start` callback hook was deprecated in v1.6 and will be removed in v1.8"
    ):
        trainer.fit(model)
    with pytest.deprecated_call(
        match="The `on_init_end` callback hook was deprecated in v1.6 and will be removed in v1.8"
    ):
        trainer.validate(model)


def test_v1_8_0_deprecated_call_hook():
    trainer = Trainer(
        max_epochs=1,
        limit_val_batches=0.1,
        limit_train_batches=0.2,
        enable_progress_bar=False,
        logger=False,
    )
    with pytest.deprecated_call(match="was deprecated in v1.6 and will be removed in v1.8."):
        trainer.call_hook("test_hook")


def test_v1_8_0_deprecated_warning_positional_category():
    with pytest.deprecated_call(match=r"use `category=FutureWarning."):
        rank_zero_warn("foo", FutureWarning)


def test_v1_8_0_deprecated_on_hpc_hooks(tmpdir):
    class TestModelSave(BoringModel):
        def on_hpc_save(self):
            print("on_hpc_save override")

    class TestModelLoad(BoringModel):
        def on_hpc_load(self):
            print("on_hpc_load override")

    save_model = TestModelSave()
    load_model = TestModelLoad()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, fast_dev_run=True)

    with pytest.deprecated_call(
        match=r"Method `LightningModule.on_hpc_save` is deprecated in v1.6 and will be removed in v1.8."
    ):
        trainer.fit(save_model)
    with pytest.deprecated_call(
        match=r"Method `LightningModule.on_hpc_load` is deprecated in v1.6 and will be removed in v1.8."
    ):
        trainer.fit(load_model)


def test_v1_8_0_deprecated_run_stage():
    trainer = Trainer()
    trainer._run_stage = Mock()
    with pytest.deprecated_call(match="`Trainer.run_stage` is deprecated in v1.6 and will be removed in v1.8."):
        trainer.run_stage()


def test_v1_8_0_trainer_verbose_evaluate():
    trainer = Trainer()
    with pytest.deprecated_call(match="verbose_evaluate` property has been deprecated and will be removed in v1.8"):
        assert trainer.verbose_evaluate

    with pytest.deprecated_call(match="verbose_evaluate` property has been deprecated and will be removed in v1.8"):
        trainer.verbose_evaluate = False


def test_v1_8_0_deprecated_trainer_should_rank_save_checkpoint(tmpdir):
    trainer = Trainer()
    with pytest.deprecated_call(
        match=r"`Trainer.should_rank_save_checkpoint` is deprecated in v1.6 and will be removed in v1.8."
    ):
        _ = trainer.should_rank_save_checkpoint


def test_v1_8_0_deprecate_trainer_callback_hook_mixin():
    methods_with_self = [
        "on_before_accelerator_backend_setup",
        "on_configure_sharded_model",
        "on_init_start",
        "on_init_end",
        "on_fit_start",
        "on_fit_end",
        "on_sanity_check_start",
        "on_sanity_check_end",
        "on_train_epoch_start",
        "on_train_epoch_end",
        "on_validation_epoch_start",
        "on_validation_epoch_end",
        "on_test_epoch_start",
        "on_test_epoch_end",
        "on_predict_epoch_start",
        "on_epoch_start",
        "on_epoch_end",
        "on_train_start",
        "on_train_end",
        "on_pretrain_routine_start",
        "on_pretrain_routine_end",
        "on_batch_start",
        "on_batch_end",
        "on_validation_start",
        "on_validation_end",
        "on_test_start",
        "on_test_end",
        "on_predict_start",
        "on_predict_end",
        "on_after_backward",
    ]
    methods_with_stage = [
        "setup",
        "teardown",
    ]
    methods_with_batch_batch_idx_dataloader_idx = [
        "on_train_batch_start",
        "on_validation_batch_start",
        "on_test_batch_start",
        "on_predict_batch_start",
    ]
    methods_with_outputs_batch_batch_idx_dataloader_idx = [
        "on_train_batch_end",
        "on_validation_batch_end",
        "on_test_batch_end",
        "on_predict_batch_end",
    ]
    methods_with_checkpoint = ["on_save_checkpoint", "on_load_checkpoint"]
    trainer = Trainer(
        max_epochs=1,
        limit_val_batches=0.1,
        limit_train_batches=0.2,
        enable_progress_bar=False,
        logger=False,
    )
    model = BoringModel()
    # need to attach model to trainer for testing of `on_pretrain_routine_start`
    trainer.fit(model)
    for method_name in methods_with_self:
        fn = getattr(trainer, method_name, None)
        with pytest.deprecated_call(match="was deprecated in v1.6 and will be removed in v1.8"):
            fn()
    for method_name in methods_with_stage:
        fn = getattr(trainer, method_name)
        with pytest.deprecated_call(match="was deprecated in v1.6 and will be removed in v1.8"):
            fn(stage="test")
    for method_name in methods_with_batch_batch_idx_dataloader_idx:
        fn = getattr(trainer, method_name)
        with pytest.deprecated_call(match="was deprecated in v1.6 and will be removed in v1.8"):
            fn(batch={}, batch_idx=0, dataloader_idx=0)
    for method_name in methods_with_outputs_batch_batch_idx_dataloader_idx:
        fn = getattr(trainer, method_name)
        with pytest.deprecated_call(match="was deprecated in v1.6 and will be removed in v1.8"):
            fn(outputs=torch.tensor([[1.0, -1.0], [1.0, -1.0]]), batch={}, batch_idx=0, dataloader_idx=0)
    for method_name in methods_with_checkpoint:
        fn = getattr(trainer, method_name)
        with pytest.deprecated_call(match="was deprecated in v1.6 and will be removed in v1.8"):
            fn(checkpoint={})
    with pytest.deprecated_call(match="was deprecated in v1.6 and will be removed in v1.8"):
        trainer.on_predict_epoch_end(outputs=torch.tensor([[1.0, -1.0], [1.0, -1.0]]))
    with pytest.deprecated_call(match="was deprecated in v1.6 and will be removed in v1.8"):
        trainer.on_exception(exception=Exception)
    with pytest.deprecated_call(match="was deprecated in v1.6 and will be removed in v1.8"):
        trainer.on_before_backward(loss=torch.tensor([[1.0, -1.0], [1.0, -1.0]]))
    with pytest.deprecated_call(match="was deprecated in v1.6 and will be removed in v1.8"):
        trainer.on_before_optimizer_step(
            optimizer=optim.SGD(model.parameters(), lr=0.01, momentum=0.9), optimizer_idx=0
        )
    with pytest.deprecated_call(match="was deprecated in v1.6 and will be removed in v1.8"):
        trainer.on_before_zero_grad(optimizer=optim.SGD(model.parameters(), lr=0.01, momentum=0.9))
