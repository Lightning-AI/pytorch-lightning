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
from re import escape
from unittest.mock import Mock

import pytest
import torch

from pytorch_lightning import Callback, Trainer
from pytorch_lightning.demos.boring_classes import BoringModel
from pytorch_lightning.overrides.distributed import IndexBatchSamplerWrapper
from pytorch_lightning.strategies import SingleDeviceStrategy
from tests_pytorch.deprecated_api import _soft_unimport_module


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


def test_v1_7_0_deprecate_lightning_distributed(tmpdir):
    with pytest.deprecated_call(match="LightningDistributed is deprecated in v1.5 and will be removed in v1.7."):
        from pytorch_lightning.distributed.dist import LightningDistributed

        _ = LightningDistributed()


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


def test_v1_7_0_deprecated_slurm_job_id():
    trainer = Trainer()
    with pytest.deprecated_call(match="Method `slurm_job_id` is deprecated in v1.6.0 and will be removed in v1.7.0."):
        trainer.slurm_job_id


def test_v1_7_0_deprecated_max_steps_none(tmpdir):
    with pytest.deprecated_call(match="`max_steps = None` is deprecated in v1.5"):
        _ = Trainer(max_steps=None)

    trainer = Trainer()
    with pytest.deprecated_call(match="`max_steps = None` is deprecated in v1.5"):
        trainer.fit_loop.max_steps = None


def test_v1_7_0_index_batch_sampler_wrapper_batch_indices():
    sampler = IndexBatchSamplerWrapper(Mock())
    with pytest.deprecated_call(match="was deprecated in v1.5 and will be removed in v1.7"):
        _ = sampler.batch_indices

    with pytest.deprecated_call(match="was deprecated in v1.5 and will be removed in v1.7"):
        sampler.batch_indices = []


def test_v1_7_0_post_dispatch_hook():
    class CustomPlugin(SingleDeviceStrategy):
        def post_dispatch(self, trainer):
            pass

    with pytest.deprecated_call(match=escape("`CustomPlugin.post_dispatch()` has been deprecated in v1.6")):
        CustomPlugin(torch.device("cpu"))
