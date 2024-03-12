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
from unittest import mock
from unittest.mock import Mock

import pytest
import torch
from lightning.fabric.utilities.imports import _TORCH_EQUAL_2_0
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.loops import _Loop
from lightning.pytorch.loops.utilities import _no_grad_context


@pytest.mark.parametrize("trainer_fn", ["validate", "test", "predict"])
def test_eval_inference_mode(tmp_path, trainer_fn):
    class BoringModelNoGrad(BoringModel):
        def assert_not_enabled(self):
            assert not torch.is_grad_enabled()
            assert not torch.is_inference_mode_enabled()

        on_test_start = assert_not_enabled
        on_validation_start = assert_not_enabled
        on_predict_start = assert_not_enabled

    class BoringModelForInferenceMode(BoringModel):
        def assert_enabled(self):
            assert not torch.is_grad_enabled()
            assert torch.is_inference_mode_enabled()

        on_test_start = assert_enabled
        on_validation_start = assert_enabled
        on_predict_start = assert_enabled

    trainer = Trainer(default_root_dir=tmp_path, logger=False, inference_mode=False, fast_dev_run=True)
    getattr(trainer, trainer_fn)(BoringModelNoGrad())
    trainer = Trainer(logger=False, inference_mode=True, fast_dev_run=True)
    getattr(trainer, trainer_fn)(BoringModelForInferenceMode())


def test_no_grad_context():
    trainer = Mock()

    class Foo:
        @_no_grad_context
        def run(self): ...

    f = Foo()
    with pytest.raises(TypeError, match="Foo` needs to be a Loop"):
        f.run()

    class Foo(_Loop):
        @_no_grad_context
        def run(self): ...

    f = Foo(trainer)
    with pytest.raises(TypeError, match="Foo.inference_mode` needs to be defined"):
        f.run()

    class Foo(_Loop):
        def __init__(self):
            super().__init__(trainer)
            self.inference_mode = False

        @_no_grad_context
        def run(self): ...

    f = Foo()
    with mock.patch("torch.no_grad") as no_grad_mock:
        f.run()
    no_grad_mock.assert_called_once_with()
    f.inference_mode = True
    with mock.patch("torch.inference_mode") as inference_mode_mock:
        f.run()
    if not _TORCH_EQUAL_2_0:
        inference_mode_mock.assert_called_once_with()
