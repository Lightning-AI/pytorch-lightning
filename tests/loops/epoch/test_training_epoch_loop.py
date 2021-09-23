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
import pytest

from pytorch_lightning.loops import TrainingEpochLoop


_out00 = {"loss": 0.0}
_out01 = {"loss": 0.1}
_out02 = {"loss": 0.2}
_out03 = {"loss": 0.3}
_out10 = {"loss": 1.0}
_out11 = {"loss": 1.1}
_out12 = {"loss": 1.2}
_out13 = {"loss": 1.3}


@pytest.mark.parametrize(
    "num_optimizers,batch_outputs,expected",
    [
        (1, [], []),
        (1, [[]], []),
        # 1 batch
        (1, [[{0: _out00}]], [_out00]),
        # 2 batches
        (1, [[{0: _out00}], [{0: _out01}]], [_out00, _out01]),
        # 1 batch, 2 optimizers
        (2, [[{0: _out00, 1: _out01}]], [_out00, _out01]),
        # 2 batches, 2 optimizers
        (
            2,
            [[{0: _out00, 1: _out01}], [{0: _out10, 1: _out11}]],
            [[_out00, _out10], [_out01, _out11]],
        ),
        # 4 batches, 2 optimizers, different frequency
        (
            2,
            [[{0: _out00}], [{1: _out10}], [{1: _out11}], [{0: _out01}]],
            [[_out00, _out01], [_out10, _out11]],
        ),
        # 1 batch, tbptt with 2 splits (uneven)
        (1, [[{0: _out00}, {0: _out01}], [{0: _out03}]], [[_out00, _out01], [_out03]]),
        # 3 batches, tbptt with 2 splits, 2 optimizers alternating
        (
            2,
            [[{0: _out00}, {0: _out01}], [{1: _out10}, {1: _out11}], [{0: _out02}, {0: _out03}]],
            [[[_out00, _out01], [], [_out02, _out03]], [[], [_out10, _out11], []]],
        ),
    ],
)
def test_prepare_outputs_training_batch_end_automatic(num_optimizers, batch_outputs, expected):
    """Test that the loop converts the nested lists of outputs to the format that the `training_epoch_end` hook
    currently expects in the case of automatic optimization."""
    prepared = TrainingEpochLoop._prepare_outputs_training_epoch_end(
        batch_outputs,
        automatic=True,
        num_optimizers=num_optimizers,
    )
    assert prepared == expected


@pytest.mark.parametrize(
    "batch_outputs,expected",
    [
        ([], []),
        ([[]], []),
        # 1 batch
        ([[_out00]], [_out00]),
        # 2 batches
        ([[_out00], [_out01]], [_out00, _out01]),
        # skipped outputs
        ([[_out00], [], [], [_out03]], [_out00, _out03]),
        # tbptt with 2 splits, uneven, skipped output
        ([[_out00, _out01], [_out02, _out03], [], [_out10]], [[_out00, _out01], [_out02, _out03], [_out10]]),
    ],
)
def test_prepare_outputs_training_batch_end_manual(batch_outputs, expected):
    """Test that the loop converts the nested lists of outputs to the format that the `training_epoch_end` hook
    currently expects in the case of manual optimization."""
    prepared = TrainingEpochLoop._prepare_outputs_training_epoch_end(
        batch_outputs,
        automatic=False,
        num_optimizers=-1,  # does not matter for manual optimization
    )
    assert prepared == expected
