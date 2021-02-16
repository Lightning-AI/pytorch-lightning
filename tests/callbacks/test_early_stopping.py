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
import os
import pickle
from unittest import mock

import cloudpickle
import numpy as np
import pytest
import torch

from pytorch_lightning import _logger, seed_everything, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.trainer.states import TrainerState
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers import BoringModel
from tests.helpers.datamodules import ClassifDataModule
from tests.helpers.simple_models import ClassificationModel


class EarlyStoppingTestRestore(EarlyStopping):
    # this class has to be defined outside the test function, otherwise we get pickle error
    def __init__(self, expected_state, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expected_state = expected_state
        # cache the state for each epoch
        self.saved_states = []

    def on_train_start(self, trainer, pl_module):
        if self.expected_state:
            assert self.on_save_checkpoint(trainer, pl_module) == self.expected_state

    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        self.saved_states.append(self.on_save_checkpoint(trainer, pl_module).copy())


def test_resume_early_stopping_from_checkpoint(tmpdir):
    """
    Prevent regressions to bugs:
    https://github.com/PyTorchLightning/pytorch-lightning/issues/1464
    https://github.com/PyTorchLightning/pytorch-lightning/issues/1463
    """
    seed_everything(42)
    model = ClassificationModel()
    dm = ClassifDataModule()
    checkpoint_callback = ModelCheckpoint(dirpath=tmpdir, monitor="train_loss", save_top_k=1)
    early_stop_callback = EarlyStoppingTestRestore(None, monitor='train_loss')
    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[early_stop_callback, checkpoint_callback],
        num_sanity_val_steps=0,
        max_epochs=4,
    )
    trainer.fit(model, datamodule=dm)

    checkpoint_filepath = checkpoint_callback.kth_best_model_path
    # ensure state is persisted properly
    checkpoint = torch.load(checkpoint_filepath)
    # the checkpoint saves "epoch + 1"
    early_stop_callback_state = early_stop_callback.saved_states[checkpoint["epoch"] - 1]
    assert 4 == len(early_stop_callback.saved_states)
    assert checkpoint["callbacks"][type(early_stop_callback)] == early_stop_callback_state

    # ensure state is reloaded properly (assertion in the callback)
    early_stop_callback = EarlyStoppingTestRestore(early_stop_callback_state, monitor='train_loss')
    new_trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        resume_from_checkpoint=checkpoint_filepath,
        callbacks=[early_stop_callback],
    )

    with pytest.raises(MisconfigurationException, match=r'.*you restored a checkpoint with current_epoch*'):
        new_trainer.fit(model)


@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
def test_early_stopping_no_extraneous_invocations(tmpdir):
    """Test to ensure that callback methods aren't being invoked outside of the callback handler."""
    model = ClassificationModel()
    dm = ClassifDataModule()
    early_stop_callback = EarlyStopping(monitor='train_loss')
    expected_count = 4
    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[early_stop_callback],
        limit_train_batches=4,
        limit_val_batches=4,
        max_epochs=expected_count,
    )
    trainer.fit(model, datamodule=dm)

    assert trainer.early_stopping_callback == early_stop_callback
    assert trainer.early_stopping_callbacks == [early_stop_callback]
    assert len(trainer.dev_debugger.early_stopping_history) == expected_count


@pytest.mark.parametrize(
    "loss_values, patience, expected_stop_epoch",
    [
        ([6, 5, 5, 5, 5, 5], 3, 4),
        ([6, 5, 4, 4, 3, 3], 1, 3),
        ([6, 5, 6, 5, 5, 5], 3, 4),
    ],
)
def test_early_stopping_patience(tmpdir, loss_values, patience, expected_stop_epoch):
    """Test to ensure that early stopping is not triggered before patience is exhausted."""

    class ModelOverrideValidationReturn(BoringModel):
        validation_return_values = torch.Tensor(loss_values)

        def validation_epoch_end(self, outputs):
            loss = self.validation_return_values[self.current_epoch]
            return {"test_val_loss": loss}

    model = ModelOverrideValidationReturn()
    early_stop_callback = EarlyStopping(monitor="test_val_loss", patience=patience, verbose=True)
    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[early_stop_callback],
        val_check_interval=1.0,
        num_sanity_val_steps=0,
        max_epochs=10,
    )
    trainer.fit(model)
    assert trainer.current_epoch == expected_stop_epoch


@pytest.mark.parametrize('validation_step', ['base', None])
@pytest.mark.parametrize(
    "loss_values, patience, expected_stop_epoch",
    [
        ([6, 5, 5, 5, 5, 5], 3, 4),
        ([6, 5, 4, 4, 3, 3], 1, 3),
        ([6, 5, 6, 5, 5, 5], 3, 4),
    ],
)
def test_early_stopping_patience_train(tmpdir, validation_step, loss_values, patience, expected_stop_epoch):
    """Test to ensure that early stopping is not triggered before patience is exhausted."""

    class ModelOverrideTrainReturn(BoringModel):
        train_return_values = torch.Tensor(loss_values)

        def training_epoch_end(self, outputs):
            loss = self.train_return_values[self.current_epoch]
            self.log('train_loss', loss)

    model = ModelOverrideTrainReturn()

    if validation_step is None:
        model.validation_step = None

    early_stop_callback = EarlyStopping(monitor="train_loss", patience=patience, verbose=True)
    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[early_stop_callback],
        num_sanity_val_steps=0,
        max_epochs=10,
    )
    trainer.fit(model)
    assert trainer.current_epoch == expected_stop_epoch


def test_pickling(tmpdir):
    early_stopping = EarlyStopping()

    early_stopping_pickled = pickle.dumps(early_stopping)
    early_stopping_loaded = pickle.loads(early_stopping_pickled)
    assert vars(early_stopping) == vars(early_stopping_loaded)

    early_stopping_pickled = cloudpickle.dumps(early_stopping)
    early_stopping_loaded = cloudpickle.loads(early_stopping_pickled)
    assert vars(early_stopping) == vars(early_stopping_loaded)


def test_early_stopping_no_val_step(tmpdir):
    """Test that early stopping callback falls back to training metrics when no validation defined."""

    model = ClassificationModel()
    dm = ClassifDataModule()
    model.validation_step = None
    model.val_dataloader = None

    stopping = EarlyStopping(monitor='train_loss', min_delta=0.1, patience=0)
    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[stopping],
        overfit_batches=0.20,
        max_epochs=10,
    )
    trainer.fit(model, datamodule=dm)

    assert trainer.state == TrainerState.FINISHED, f"Training failed with {trainer.state}"
    assert trainer.current_epoch < trainer.max_epochs - 1


def test_early_stopping_functionality(tmpdir):

    class CurrentModel(BoringModel):

        def validation_epoch_end(self, outputs):
            losses = [8, 4, 2, 3, 4, 5, 8, 10]
            val_loss = losses[self.current_epoch]
            self.log('abc', torch.tensor(val_loss))

    model = CurrentModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[EarlyStopping(monitor='abc')],
        overfit_batches=0.20,
        max_epochs=20,
    )
    trainer.fit(model)
    assert trainer.current_epoch == 5, 'early_stopping failed'


def test_early_stopping_functionality_arbitrary_key(tmpdir):
    """Tests whether early stopping works with a custom key and dictionary results on val step."""

    class CurrentModel(BoringModel):

        def validation_epoch_end(self, outputs):
            losses = [8, 4, 2, 3, 4, 5, 8, 10]
            val_loss = losses[self.current_epoch]
            return {'jiraffe': torch.tensor(val_loss)}

    model = CurrentModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[EarlyStopping(monitor='jiraffe')],
        overfit_batches=0.20,
        max_epochs=20,
    )
    trainer.fit(model)
    assert trainer.current_epoch >= 5, 'early_stopping failed'


@pytest.mark.parametrize('step_freeze, min_steps, min_epochs', [(5, 1, 1), (5, 1, 3), (3, 15, 1)])
def test_min_steps_override_early_stopping_functionality(tmpdir, step_freeze, min_steps, min_epochs):
    """Excepted Behaviour:
    IF `min_steps` was set to a higher value than the `trainer.global_step` when `early_stopping` is being triggered,
    THEN the trainer should continue until reaching `trainer.global_step` == `min_steps`, and stop.

    IF `min_epochs` resulted in a higher number of steps than the `trainer.global_step`
        when `early_stopping` is being triggered,
    THEN the trainer should continue until reaching
        `trainer.global_step` == `min_epochs * len(train_dataloader)`, and stop.
    This test validate this expected behaviour

    IF both `min_epochs` and `min_steps` are provided and higher than the `trainer.global_step`
        when `early_stopping` is being triggered,
    THEN the highest between `min_epochs * len(train_dataloader)` and `min_steps` would be reached.

    Caviat: IF min_steps is divisible by len(train_dataloader), then it will do min_steps + len(train_dataloader)

    This test validate those expected behaviours
    """

    _logger.disabled = True

    original_loss_value = 10
    limit_train_batches = 3
    patience = 3

    class Model(BoringModel):

        def __init__(self, step_freeze):
            super(Model, self).__init__()

            self._step_freeze = step_freeze

            self._loss_value = 10.0
            self._eps = 1e-1
            self._count_decrease = 0
            self._values = []

        def training_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            return {"loss": loss}

        def validation_step(self, batch, batch_idx):
            return {"test_val_loss": self._loss_value}

        def validation_epoch_end(self, outputs):
            _mean = np.mean([x['test_val_loss'] for x in outputs])
            if self.trainer.global_step <= self._step_freeze:
                self._count_decrease += 1
                self._loss_value -= self._eps
            self._values.append(_mean)
            return {"test_val_loss": _mean}

    model = Model(step_freeze)
    model.training_step_end = None
    model.test_dataloader = None
    early_stop_callback = EarlyStopping(monitor="test_val_loss", patience=patience, verbose=True)
    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[early_stop_callback],
        limit_train_batches=limit_train_batches,
        limit_val_batches=2,
        min_steps=min_steps,
        min_epochs=min_epochs
    )
    trainer.fit(model)

    # Make sure loss was properly decreased
    assert abs(original_loss_value - (model._count_decrease) * model._eps - model._loss_value) < 1e-6

    pos_diff = (np.diff(model._values) == 0).nonzero()[0][0]

    # Compute when the latest validation epoch end happened
    latest_validation_epoch_end = (pos_diff // limit_train_batches) * limit_train_batches
    if pos_diff % limit_train_batches == 0:
        latest_validation_epoch_end += limit_train_batches

    # Compute early stopping latest step
    by_early_stopping = latest_validation_epoch_end + (1 + limit_train_batches) * patience

    # Compute min_epochs latest step
    by_min_epochs = min_epochs * limit_train_batches

    # Make sure the trainer stops for the max of all minimun requirements
    assert trainer.global_step == max(min_steps, by_early_stopping, by_min_epochs), \
        (trainer.global_step, max(min_steps, by_early_stopping, by_min_epochs), step_freeze, min_steps, min_epochs)

    _logger.disabled = False


def test_early_stopping_mode_options():
    with pytest.raises(MisconfigurationException, match="`mode` can be auto, .* got unknown_option"):
        EarlyStopping(mode="unknown_option")
