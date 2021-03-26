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
"""
Tests to ensure that the training loop works with a dict
"""
from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
from tests.helpers.deterministic_model import DeterministicModel


def test_validation_step_no_return(tmpdir):
    """
    Test that val step can return nothing
    """

    class TestModel(DeterministicModel):

        def backward(self, loss, optimizer, optimizer_idx):
            return LightningModule.backward(self, loss, optimizer, optimizer_idx)

    model = TestModel()
    model.training_step = model.training_step__dict_return
    model.validation_step = model.validation_step__no_return
    model.validation_step_end = None
    model.validation_epoch_end = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        weights_summary=None,
    )
    trainer.fit(model)

    # out are the results of the full loop
    # eval_results are output of _evaluate
    out, eval_results = trainer.run_evaluation()
    assert len(out) == 1
    assert len(eval_results) == 0

    # make sure correct steps were called
    assert model.validation_step_called
    assert not model.validation_step_end_called
    assert not model.validation_epoch_end_called


def test_validation_step_scalar_return(tmpdir):
    """
    Test that val step can return a scalar
    """
    model = DeterministicModel()
    model.training_step = model.training_step__dict_return
    model.validation_step = model.validation_step__scalar_return
    model.validation_step_end = None
    model.validation_epoch_end = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        weights_summary=None,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=2,
    )
    trainer.fit(model)

    # out are the results of the full loop
    # eval_results are output of _evaluate
    out, eval_results = trainer.run_evaluation()
    assert len(out) == 1
    assert len(eval_results) == 2
    assert eval_results[0] == 171 and eval_results[1] == 171

    # make sure correct steps were called
    assert model.validation_step_called
    assert not model.validation_step_end_called
    assert not model.validation_epoch_end_called


def test_validation_step_arbitrary_dict_return(tmpdir):
    """
    Test that val step can return an arbitrary dict
    """
    model = DeterministicModel()
    model.training_step = model.training_step__dict_return
    model.validation_step = model.validation_step__dummy_dict_return
    model.validation_step_end = None
    model.validation_epoch_end = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        weights_summary=None,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=2,
    )
    trainer.fit(model)

    # out are the results of the full loop
    # eval_results are output of _evaluate
    callback_metrics, eval_results = trainer.run_evaluation()
    assert len(callback_metrics) == 1
    assert len(eval_results) == 2
    assert eval_results[0]['some'] == 171
    assert eval_results[1]['some'] == 171

    assert eval_results[0]['value'] == 'a'
    assert eval_results[1]['value'] == 'a'

    # make sure correct steps were called
    assert model.validation_step_called
    assert not model.validation_step_end_called
    assert not model.validation_epoch_end_called


def test_val_step_step_end_no_return(tmpdir):
    """
    Test that val step + val step end work (with no return in val step end)
    """

    model = DeterministicModel()
    model.training_step = model.training_step__dict_return
    model.validation_step = model.validation_step__dict_return
    model.validation_step_end = model.validation_step_end__no_return
    model.validation_epoch_end = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        weights_summary=None,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=2,
    )
    trainer.fit(model)

    # out are the results of the full loop
    # eval_results are output of _evaluate
    callback_metrics, eval_results = trainer.run_evaluation()
    assert len(callback_metrics) == 1
    assert len(eval_results) == 0

    # make sure correct steps were called
    assert model.validation_step_called
    assert model.validation_step_end_called
    assert not model.validation_epoch_end_called
