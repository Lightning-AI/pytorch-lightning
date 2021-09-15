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
from unittest.mock import Mock

import pytest
import torch
from torch.optim import Adam, SGD

from pytorch_lightning import Trainer
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.loops.optimization.optimizer_loop import ClosureResult
from tests.helpers import BoringModel


def test_closure_result_deepcopy():
    closure_loss = torch.tensor(123.45)
    result = ClosureResult(closure_loss)

    assert closure_loss.data_ptr() == result.closure_loss.data_ptr()
    # the `loss` is cloned so the storage is different
    assert closure_loss.data_ptr() != result.loss.data_ptr()

    copy = result.drop_closure_loss()
    assert result.loss == copy.loss
    assert copy.closure_loss is None

    # no copy
    assert id(result.loss) == id(copy.loss)
    assert result.loss.data_ptr() == copy.loss.data_ptr()


def test_closure_result_apply_accumulation():
    closure_loss = torch.tensor(25.0)
    result = ClosureResult.from_training_step_output(closure_loss, 5)
    assert result.loss == 5


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
    model.optimizer_step = Mock(wraps=model.optimizer_step)
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=10,
        progress_bar_refresh_rate=0,
    )
    trainer.fit(model)

    positional_args = [c[0] for c in model.optimizer_step.call_args_list]
    pl_optimizer_sequence = [args[2] for args in positional_args]
    opt_idx_sequence = [args[3] for args in positional_args]
    assert all(isinstance(opt, LightningOptimizer) for opt in pl_optimizer_sequence)
    optimizer_sequence = [opt._optimizer.__class__.__name__ for opt in pl_optimizer_sequence]
    assert list(zip(opt_idx_sequence, optimizer_sequence)) == expected
