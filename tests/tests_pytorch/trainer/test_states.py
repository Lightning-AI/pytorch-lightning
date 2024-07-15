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
from lightning.pytorch.trainer.states import RunningStage, TrainerFn, TrainerState, TrainerStatus


def test_initialize_state():
    """Tests that state is INITIALIZING after Trainer creation."""
    trainer = Trainer()
    assert trainer.state == TrainerState(status=TrainerStatus.INITIALIZING, fn=None, stage=None)


@pytest.mark.parametrize(
    "extra_params",
    [pytest.param({"fast_dev_run": True}, id="Fast-Run"), pytest.param({"max_steps": 1}, id="Single-Step")],
)
def test_trainer_fn_while_running(tmp_path, extra_params):
    class TestModel(BoringModel):
        def __init__(self, expected_fn, expected_stage):
            super().__init__()
            self.expected_fn = expected_fn
            self.expected_stage = expected_stage
            self.lr = 0.1

        def on_train_batch_start(self, *_):
            assert self.trainer.state.status == TrainerStatus.RUNNING
            assert self.trainer.state.fn == self.expected_fn
            assert self.trainer.training

        def on_sanity_check_start(self, *_):
            assert self.trainer.state.status == TrainerStatus.RUNNING
            assert self.trainer.state.fn == self.expected_fn
            assert self.trainer.sanity_checking

        def on_validation_batch_start(self, *_):
            assert self.trainer.state.status == TrainerStatus.RUNNING
            assert self.trainer.state.fn == self.expected_fn
            assert self.trainer.validating or self.trainer.sanity_checking

        def on_test_batch_start(self, *_):
            assert self.trainer.state.status == TrainerStatus.RUNNING
            assert self.trainer.state.fn == self.expected_fn
            assert self.trainer.testing

    trainer = Trainer(default_root_dir=tmp_path, **extra_params)

    model = TestModel(TrainerFn.FITTING, RunningStage.TRAINING)
    trainer.fit(model)
    assert trainer.state.finished

    model = TestModel(TrainerFn.VALIDATING, RunningStage.VALIDATING)
    trainer.validate(model)
    assert trainer.state.finished

    model = TestModel(TrainerFn.TESTING, RunningStage.TESTING)
    trainer.test(model)
    assert trainer.state.finished


@pytest.mark.parametrize(
    "extra_params",
    [pytest.param({"fast_dev_run": True}, id="Fast-Run"), pytest.param({"max_steps": 1}, id="Single-Step")],
)
def test_interrupt_state_on_keyboard_interrupt(tmp_path, extra_params):
    """Tests that state is set to INTERRUPTED on KeyboardInterrupt."""
    model = BoringModel()

    class InterruptCallback(Callback):
        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            raise KeyboardInterrupt

    trainer = Trainer(callbacks=[InterruptCallback()], default_root_dir=tmp_path, **extra_params)

    with pytest.raises(SystemExit):
        trainer.fit(model)
    assert trainer.interrupted
