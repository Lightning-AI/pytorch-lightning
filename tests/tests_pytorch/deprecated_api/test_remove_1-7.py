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

import pytest
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.demos.boring_classes import BoringModel


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


def test_v1_7_0_deprecated_max_steps_none(tmpdir):
    with pytest.deprecated_call(match="`max_steps = None` is deprecated in v1.5"):
        _ = Trainer(max_steps=None)

    trainer = Trainer()
    with pytest.deprecated_call(match="`max_steps = None` is deprecated in v1.5"):
        trainer.fit_loop.max_steps = None


def test_v1_7_0_post_dispatch_hook():
    class CustomPlugin(SingleDeviceStrategy):
        def post_dispatch(self, trainer):
            pass

    with pytest.deprecated_call(match=escape("`CustomPlugin.post_dispatch()` has been deprecated in v1.6")):
        CustomPlugin(torch.device("cpu"))
