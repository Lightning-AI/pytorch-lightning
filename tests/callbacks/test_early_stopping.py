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
from copy import deepcopy
import pickle

import cloudpickle
import pytest
import torch

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from tests.base import EvalModelTemplate
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class EarlyStoppingTestRestore(EarlyStopping):
    # this class has to be defined outside the test function, otherwise we get pickle error
    def __init__(self, expected_state=None):
        super().__init__()
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
    model = EvalModelTemplate()
    checkpoint_callback = ModelCheckpoint(dirpath=tmpdir, monitor="early_stop_on", save_top_k=1)
    early_stop_callback = EarlyStoppingTestRestore()
    trainer = Trainer(
        default_root_dir=tmpdir,
        checkpoint_callback=checkpoint_callback,
        callbacks=[early_stop_callback],
        num_sanity_val_steps=0,
        max_epochs=4,
    )
    trainer.fit(model)

    checkpoint_filepath = checkpoint_callback.kth_best_model_path
    # ensure state is persisted properly
    checkpoint = torch.load(checkpoint_filepath)
    # the checkpoint saves "epoch + 1"
    early_stop_callback_state = early_stop_callback.saved_states[checkpoint["epoch"] - 1]
    assert 4 == len(early_stop_callback.saved_states)
    assert checkpoint["callbacks"][type(early_stop_callback)] == early_stop_callback_state

    # ensure state is reloaded properly (assertion in the callback)
    early_stop_callback = EarlyStoppingTestRestore(early_stop_callback_state)
    new_trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        resume_from_checkpoint=checkpoint_filepath,
        callbacks=[early_stop_callback],
    )

    with pytest.raises(MisconfigurationException, match=r'.*you restored a checkpoint with current_epoch*'):
        new_trainer.fit(model)


def test_early_stopping_no_extraneous_invocations(tmpdir):
    """Test to ensure that callback methods aren't being invoked outside of the callback handler."""
    os.environ['PL_DEV_DEBUG'] = '1'

    model = EvalModelTemplate()
    expected_count = 4
    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[EarlyStopping()],
        val_check_interval=1.0,
        max_epochs=expected_count,
    )
    trainer.fit(model)

    assert len(trainer.dev_debugger.early_stopping_history) == expected_count


@pytest.mark.parametrize(
    "loss_values, patience, expected_stop_epoch",
    [([6, 5, 5, 5, 5, 5], 3, 4), ([6, 5, 4, 4, 3, 3], 1, 3), ([6, 5, 6, 5, 5, 5], 3, 4),],
)
def test_early_stopping_patience(tmpdir, loss_values, patience, expected_stop_epoch):
    """Test to ensure that early stopping is not triggered before patience is exhausted."""

    class ModelOverrideValidationReturn(EvalModelTemplate):
        validation_return_values = torch.Tensor(loss_values)
        count = 0

        def validation_epoch_end(self, outputs):
            loss = self.validation_return_values[self.count]
            self.count += 1
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

    class CurrentModel(EvalModelTemplate):
        def training_step(self, *args, **kwargs):
            output = super().training_step(*args, **kwargs)
            output.update({'my_train_metric': output['loss']})  # could be anything else
            return output

    model = CurrentModel()
    model.validation_step = None
    model.val_dataloader = None

    stopping = EarlyStopping(monitor='my_train_metric', min_delta=0.1, patience=0)
    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[stopping],
        overfit_batches=0.20,
        max_epochs=10,
    )
    result = trainer.fit(model)

    assert result == 1, 'training failed to complete'
    assert trainer.current_epoch < trainer.max_epochs - 1


def test_early_stopping_functionality(tmpdir):

    class CurrentModel(EvalModelTemplate):
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

    class CurrentModel(EvalModelTemplate):
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
