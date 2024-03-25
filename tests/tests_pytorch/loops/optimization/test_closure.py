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
import pytest
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.utilities.exceptions import MisconfigurationException


def test_optimizer_step_no_closure_raises(tmp_path):
    class TestModel(BoringModel):
        def optimizer_step(self, epoch=None, batch_idx=None, optimizer=None, optimizer_closure=None, **_):
            # does not call `optimizer_closure()`
            pass

    model = TestModel()
    trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=1)
    with pytest.raises(MisconfigurationException, match="The closure hasn't been executed"):
        trainer.fit(model)

    class TestModel(BoringModel):
        def configure_optimizers(self):
            class BrokenSGD(torch.optim.SGD):
                def step(self, closure=None):
                    # forgot to pass the closure
                    return super().step()

            return BrokenSGD(self.layer.parameters(), lr=0.1)

    model = TestModel()
    trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=1)
    with pytest.raises(MisconfigurationException, match="The closure hasn't been executed"):
        trainer.fit(model)


def test_closure_with_no_grad_optimizer(tmp_path):
    """Test that the closure is guaranteed to run with grad enabled.

    There are certain third-party library optimizers
    (such as Hugging Face Transformers' AdamW) that set `no_grad` during the `step` operation.

    """

    class NoGradAdamW(torch.optim.AdamW):
        @torch.no_grad()
        def step(self, closure):
            if closure is not None:
                closure()
            return super().step()

    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx):
            assert torch.is_grad_enabled()
            return super().training_step(batch, batch_idx)

        def configure_optimizers(self):
            return NoGradAdamW(self.parameters(), lr=0.1)

    trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=1)
    model = TestModel()
    trainer.fit(model)
