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
from unittest import mock
from unittest.mock import Mock

import pytest
import torch
from torch.optim import Adam, SGD

from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.loops.optimization.optimizer_loop import ClosureResult
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers import BoringModel


def test_closure_result_deepcopy():
    closure_loss = torch.tensor(123.45)
    result = ClosureResult(closure_loss)

    assert closure_loss.data_ptr() == result.closure_loss.data_ptr()
    # the `loss` is cloned so the storage is different
    assert closure_loss.data_ptr() != result.loss.data_ptr()

    copy = result.asdict()
    assert result.loss == copy["loss"]
    assert copy.keys() == {"loss"}

    # no copy
    assert id(result.loss) == id(copy["loss"])
    assert result.loss.data_ptr() == copy["loss"].data_ptr()


def test_closure_result_apply_accumulation():
    closure_loss = torch.tensor(25.0)
    result = ClosureResult.from_training_step_output(closure_loss, 5)
    assert result.loss == 5


@pytest.mark.parametrize(
    "case", [(5.0, "must return a Tensor, a dict, or None"), ({"a": 5}, "the 'loss' key needs to be present")]
)
def test_warning_invalid_trainstep_output(tmpdir, case):
    output, match = case

    class InvalidTrainStepModel(BoringModel):
        def training_step(self, batch, batch_idx):
            return output

    model = InvalidTrainStepModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=1)

    with pytest.raises(MisconfigurationException, match=match):
        trainer.fit(model)


@pytest.mark.parametrize(
    "frequencies,expected",
    [
        (
            (3, 1),
            [
                (0, "SGD"),
                (0, "SGD"),
                (0, "SGD"),
                (1, "Adam"),
                (0, "SGD"),
                (0, "SGD"),
                (0, "SGD"),
                (1, "Adam"),
                (0, "SGD"),
                (0, "SGD"),
            ],
        ),
        (
            (1, 2),
            [
                (0, "SGD"),
                (1, "Adam"),
                (1, "Adam"),
                (0, "SGD"),
                (1, "Adam"),
                (1, "Adam"),
                (0, "SGD"),
                (1, "Adam"),
                (1, "Adam"),
                (0, "SGD"),
            ],
        ),
    ],
)
def test_optimizer_frequencies(tmpdir, frequencies, expected):
    """Test that the optimizer loop runs optimization for the correct optimizer and optimizer idx when different
    frequencies are requested."""

    class CurrentModel(BoringModel):
        def training_step(self, batch, batch_idx, optimizer_idx):
            return super().training_step(batch, batch_idx)

        def configure_optimizers(self):
            opt0 = SGD(self.parameters(), lr=0.1)
            opt1 = Adam(self.parameters(), lr=0.1)
            return {"optimizer": opt0, "frequency": frequencies[0]}, {"optimizer": opt1, "frequency": frequencies[1]}

    model = CurrentModel()
    model.training_epoch_end = None
    model.optimizer_step = Mock(wraps=model.optimizer_step)
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=10,
        enable_progress_bar=False,
    )
    trainer.fit(model)

    positional_args = [c[0] for c in model.optimizer_step.call_args_list]
    pl_optimizer_sequence = [args[2] for args in positional_args]
    opt_idx_sequence = [args[3] for args in positional_args]
    assert all(isinstance(opt, LightningOptimizer) for opt in pl_optimizer_sequence)
    optimizer_sequence = [opt._optimizer.__class__.__name__ for opt in pl_optimizer_sequence]
    assert list(zip(opt_idx_sequence, optimizer_sequence)) == expected


class CustomException(Exception):
    pass


@mock.patch.dict(os.environ, {"PL_FAULT_TOLERANT_TRAINING": "1"})
@pytest.mark.parametrize("stop_epoch", (0, 1))
@pytest.mark.parametrize("stop_batch", (0, 1, 2))
@pytest.mark.parametrize("n_optimizers,stop_optimizer", [(2, 0), (2, 1), (3, 2)])
def test_loop_restart_progress_multiple_optimizers(tmpdir, n_optimizers, stop_optimizer, stop_epoch, stop_batch):
    """Test that Lightning can resume from a point where a training_step failed while in the middle of processing
    several optimizer steps for one batch.

    The test asserts that we end up with the same trained weights as if no failure occurred.
    """

    n_batches = 3
    n_epochs = 2

    def _assert_optimizer_sequence(method_mock, expected):
        positional_args = [c[0] for c in method_mock.call_args_list]
        sequence = [arg[3] for arg in positional_args]
        assert sequence == expected

    num_optimizers_incomplete = stop_epoch * n_batches * n_optimizers + stop_batch * n_optimizers + stop_optimizer

    opt_idx_sequence_complete = list(range(n_optimizers)) * n_epochs * n_batches  # [0, 1, 2, 0, 1, 2, 0, 1, ...]
    # +1 because we fail inside the closure inside optimizer_step()
    opt_idx_sequence_incomplete = opt_idx_sequence_complete[: (num_optimizers_incomplete + 1)]
    opt_idx_sequence_resumed = opt_idx_sequence_complete[num_optimizers_incomplete:]

    class MultipleOptimizerModel(BoringModel):
        def training_step(self, batch, batch_idx, optimizer_idx):
            if (
                fail
                and self.current_epoch == stop_epoch
                and batch_idx == stop_batch
                and optimizer_idx == stop_optimizer
            ):
                raise CustomException
            return super().training_step(batch, batch_idx)

        def configure_optimizers(self):
            return [torch.optim.SGD(self.parameters(), lr=0.1) for _ in range(n_optimizers)]

    # run without a failure, collect weights
    fail = False
    seed_everything(0)
    model = MultipleOptimizerModel()
    model.training_epoch_end = None
    model.optimizer_step = Mock(wraps=model.optimizer_step)
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=n_epochs,
        limit_train_batches=n_batches,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model)
    weights_complete = model.parameters()
    _assert_optimizer_sequence(model.optimizer_step, opt_idx_sequence_complete)

    # simulate a failure
    fail = True
    seed_everything(0)
    model = MultipleOptimizerModel()
    model.training_epoch_end = None
    model.optimizer_step = Mock(wraps=model.optimizer_step)
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=n_epochs,
        limit_train_batches=n_batches,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
    )
    with pytest.raises(CustomException):
        trainer.fit(model)

    _assert_optimizer_sequence(model.optimizer_step, opt_idx_sequence_incomplete)

    # resume from failure and collect weights
    fail = False
    seed_everything(0)
    model = MultipleOptimizerModel()
    model.training_epoch_end = None
    model.optimizer_step = Mock(wraps=model.optimizer_step)
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=n_epochs,
        limit_train_batches=n_batches,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, ckpt_path=str(tmpdir / ".pl_auto_save.ckpt"))
    weights_resumed = model.parameters()

    # check that the final weights of a resumed run match the weights of a run that never failed
    for w0, w1 in zip(weights_complete, weights_resumed):
        assert torch.allclose(w0, w1)

    _assert_optimizer_sequence(model.optimizer_step, opt_idx_sequence_resumed)
