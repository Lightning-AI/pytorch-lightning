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

from lightning.pytorch import Callback, Trainer
from lightning.pytorch.demos.boring_classes import BoringModel


@pytest.mark.parametrize("single_cb", [False, True])
def test_train_step_no_return(tmp_path, single_cb: bool):
    """Tests that only training_step can be used."""

    class CB(Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, *_):
            assert "loss" in outputs

        def on_validation_batch_end(self, trainer, pl_module, outputs, *_):
            assert "x" in outputs

        def on_test_batch_end(self, trainer, pl_module, outputs, *_):
            assert "x" in outputs

    class TestModel(BoringModel):
        def on_train_batch_end(self, outputs, *_):
            assert "loss" in outputs

        def on_validation_batch_end(self, outputs, *_):
            assert "x" in outputs

        def on_test_batch_end(self, outputs, *_):
            assert "x" in outputs

    model = TestModel()

    trainer = Trainer(
        callbacks=CB() if single_cb else [CB()],
        default_root_dir=tmp_path,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
        enable_model_summary=False,
    )

    assert any(isinstance(c, CB) for c in trainer.callbacks)

    trainer.fit(model)


def test_on_before_optimizer_setup_is_called_in_correct_order(tmp_path):
    """Ensure `on_before_optimizer_setup` runs after `configure_model` but before `configure_optimizers`."""

    order = []

    class TestCallback(Callback):
        def setup(self, trainer, pl_module, stage=None):
            order.append("setup")
            assert pl_module.layer is None
            assert len(trainer.optimizers) == 0

        def on_before_optimizer_setup(self, trainer, pl_module):
            order.append("on_before_optimizer_setup")
            # configure_model should already have been called
            assert pl_module.layer is not None
            # but optimizers are not yet created
            assert len(trainer.optimizers) == 0

        def on_fit_start(self, trainer, pl_module):
            order.append("on_fit_start")
            # optimizers should now exist
            assert len(trainer.optimizers) == 1
            assert pl_module.layer is not None

    class DemoModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.layer = None

        def configure_model(self):
            from torch import nn

            self.layer = nn.Linear(32, 2)

    model = DemoModel()

    trainer = Trainer(
        callbacks=TestCallback(),
        default_root_dir=tmp_path,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=1,
        enable_model_summary=False,
        log_every_n_steps=1,
    )

    trainer.fit(model)

    # Verify call order
    assert order == ["setup", "on_before_optimizer_setup", "on_fit_start"]
