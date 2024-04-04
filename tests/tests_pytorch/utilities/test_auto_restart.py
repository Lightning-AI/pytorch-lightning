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
import inspect

import pytest
from lightning.fabric.utilities.seed import seed_everything
from lightning.pytorch import Callback, Trainer
from lightning.pytorch.callbacks import OnExceptionCheckpoint
from lightning.pytorch.demos.boring_classes import BoringModel, RandomDataset
from lightning.pytorch.utilities.exceptions import SIGTERMException
from torch.utils.data.dataloader import DataLoader

from tests_pytorch.helpers.runif import RunIf


class TestAutoRestartModelUnderSignal(BoringModel):
    def __init__(self, should_signal: bool, failure_on_step: bool, failure_on_training: bool, on_last_batch: bool):
        super().__init__()
        self.should_signal = should_signal
        self.failure_on_step = failure_on_step
        self.failure_on_training = failure_on_training
        self.on_last_batch = on_last_batch
        self.seen_train_batches = []

    def _signal(self):
        if self.should_signal:
            # simulate `os.kill(os.getpid(), signal.SIGTERM)`
            self.trainer._signal_connector.received_sigterm = True

    def training_step(self, batch, batch_idx):
        self.seen_train_batches.append(batch)
        should_signal = self.trainer.fit_loop.epoch_loop._is_training_done if self.on_last_batch else batch_idx == 2
        if self.failure_on_step and self.failure_on_training and should_signal:
            self._signal()
        return super().training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        should_signal = (
            self.trainer.fit_loop.epoch_loop.val_loop.batch_progress.is_last_batch
            if self.on_last_batch
            else batch_idx == 2
        )
        if self.failure_on_step and not self.failure_on_training and should_signal:
            self._signal()
        return super().validation_step(batch, batch_idx)

    def on_train_epoch_end(self):
        if not self.failure_on_step and self.failure_on_training:
            self._signal()

    def on_validation_epoch_end(self):
        if not self.failure_on_step and not self.failure_on_training:
            self._signal()

    def train_dataloader(self):
        return DataLoader(RandomDataset(32, 4))

    def val_dataloader(self):
        return DataLoader(RandomDataset(32, 4))


def _fit_model(
    tmp_path, should_signal, val_check_interval, failure_on_step, failure_on_training, on_last_batch, status=None
):
    seed_everything(42)
    model = TestAutoRestartModelUnderSignal(should_signal, failure_on_step, failure_on_training, on_last_batch)

    class MyTestCallback(Callback):
        raising_function = None

        def on_exception(self, trainer, pl_module, exception):
            if isinstance(exception, SIGTERMException):
                caller = inspect.trace()[-1]
                class_name = caller[0].f_locals["self"].__class__.__name__
                self.raising_method = f"{class_name}:{caller.function}"

    test_callback = MyTestCallback()
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        limit_train_batches=4,
        limit_val_batches=4,
        val_check_interval=val_check_interval,
        num_sanity_val_steps=0,
        callbacks=[test_callback, OnExceptionCheckpoint(tmp_path)],
    )
    if should_signal:
        with pytest.raises(SIGTERMException):
            trainer.fit(model)
        assert test_callback.raising_method == status
    else:
        trainer.fit(model)
    assert trainer.received_sigterm == should_signal

    return model


@pytest.mark.parametrize("on_last_batch", [False, True])
@pytest.mark.parametrize("val_check_interval", [0.5, 1.0])
@pytest.mark.parametrize("failure_on_training", [False, True])
@pytest.mark.parametrize("failure_on_step", [False, True])
@RunIf(skip_windows=True)
def test_auto_restart_under_signal(on_last_batch, val_check_interval, failure_on_training, failure_on_step, tmp_path):
    if failure_on_step:
        if on_last_batch:
            if failure_on_training:
                # Breaking on first validation batch.
                # This is done to capture the random state of the validation dataloader.
                status = "_EvaluationLoop:_evaluation_step"
            else:
                # when breaking on last batch of validation, we should exist on `run_end` val_check_interval == 1.0
                status = "_FitLoop:on_advance_end" if val_check_interval == 1.0 else "_TrainingEpochLoop:on_advance_end"
        else:
            status = "_TrainingEpochLoop:on_advance_end" if failure_on_training else "_EvaluationLoop:_evaluation_step"
    else:
        if val_check_interval == 1.0:
            status = "_FitLoop:on_advance_end"
        else:
            # `on_train_epoch_end` happens after `on_validation_epoch_end` since Lightning v1.4
            status = "_FitLoop:on_advance_end" if failure_on_training else "_TrainingEpochLoop:on_advance_end"

    _fit_model(tmp_path, True, val_check_interval, failure_on_step, failure_on_training, on_last_batch, status=status)
