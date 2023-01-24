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
import logging
from unittest.mock import patch

import pytest

from pytorch_lightning import LightningModule
from pytorch_lightning.demos.boring_classes import BoringModel
from pytorch_lightning.loops import _TrainingEpochLoop
from pytorch_lightning.trainer.trainer import Trainer

_out00 = {"loss": 0.0}
_out01 = {"loss": 0.1}
_out02 = {"loss": 0.2}
_out03 = {"loss": 0.3}
_out10 = {"loss": 1.0}
_out11 = {"loss": 1.1}
_out12 = {"loss": 1.2}
_out13 = {"loss": 1.3}


class TestPrepareOutputs:
    def prepare_outputs(self, fn, batch_outputs, num_optimizers, automatic_optimization):
        lightning_module = LightningModule()
        lightning_module.automatic_optimization = automatic_optimization
        return fn(
            batch_outputs,
            lightning_module=lightning_module,
            num_optimizers=num_optimizers,  # does not matter for manual optimization
        )

    def prepare_outputs_training_epoch_end(self, batch_outputs, num_optimizers, automatic_optimization=True):
        return self.prepare_outputs(
            _TrainingEpochLoop._prepare_outputs_training_epoch_end,
            batch_outputs,
            num_optimizers,
            automatic_optimization=automatic_optimization,
        )

    def prepare_outputs_training_batch_end(self, batch_outputs, num_optimizers, automatic_optimization=True):
        return self.prepare_outputs(
            _TrainingEpochLoop._prepare_outputs_training_batch_end,
            batch_outputs,
            num_optimizers,
            automatic_optimization=automatic_optimization,
        )

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
            (2, [[{0: _out00, 1: _out01}], [{0: _out10, 1: _out11}]], [[_out00, _out01], [_out10, _out11]]),
            # 4 batches, 2 optimizers, different frequency
            (
                2,
                [[{0: _out00}], [{1: _out10}], [{1: _out11}], [{0: _out01}]],
                [[_out00], [_out10], [_out11], [_out01]],
            ),
        ],
    )
    def test_prepare_outputs_training_epoch_end_automatic(self, num_optimizers, batch_outputs, expected):
        """Test that the loop converts the nested lists of outputs to the format that the `training_epoch_end` hook
        currently expects in the case of automatic optimization."""
        assert self.prepare_outputs_training_epoch_end(batch_outputs, num_optimizers) == expected

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
        ],
    )
    def test_prepare_outputs_training_epoch_end_manual(self, batch_outputs, expected):
        """Test that the loop converts the nested lists of outputs to the format that the `training_epoch_end` hook
        currently expects in the case of manual optimization."""
        assert self.prepare_outputs_training_epoch_end(batch_outputs, -1, automatic_optimization=False) == expected

    @pytest.mark.parametrize(
        "num_optimizers,batch_end_outputs,expected",
        [
            (1, [], []),
            (1, [[]], []),
            # 1 optimizer
            (1, [{0: _out00}], _out00),
            # 2 optimizers
            (2, [{0: _out00, 1: _out01}], [_out00, _out01]),
        ],
    )
    def test_prepare_outputs_training_batch_end_automatic(self, num_optimizers, batch_end_outputs, expected):
        """Test that the loop converts the nested lists of outputs to the format that the `on_train_batch_end` hook
        currently expects in the case of automatic optimization."""

        assert self.prepare_outputs_training_batch_end(batch_end_outputs, num_optimizers) == expected

    @pytest.mark.parametrize(
        "batch_end_outputs,expected",
        [
            ([], []),
            ([[]], []),
            # skipped outputs
            ([_out00, None, _out02], [_out00, _out02]),
        ],
    )
    def test_prepare_outputs_training_batch_end_manual(self, batch_end_outputs, expected):
        """Test that the loop converts the nested lists of outputs to the format that the `on_train_batch_end` hook
        currently expects in the case of manual optimization."""
        assert self.prepare_outputs_training_batch_end(batch_end_outputs, -1, automatic_optimization=False) == expected


def test_no_val_on_train_epoch_loop_restart(tmpdir):
    """Test that training validation loop doesn't get triggered at the beginning of a restart."""
    trainer_kwargs = {
        "max_epochs": 1,
        "limit_train_batches": 1,
        "limit_val_batches": 1,
        "num_sanity_val_steps": 0,
        "enable_checkpointing": False,
    }
    trainer = Trainer(**trainer_kwargs)
    model = BoringModel()
    trainer.fit(model)
    ckpt_path = str(tmpdir / "last.ckpt")
    trainer.save_checkpoint(ckpt_path)

    trainer_kwargs["max_epochs"] = 2
    trainer = Trainer(**trainer_kwargs)

    with patch.object(
        trainer.fit_loop.epoch_loop.val_loop, "advance", wraps=trainer.fit_loop.epoch_loop.val_loop.advance
    ) as advance_mocked:
        trainer.fit(model, ckpt_path=ckpt_path)
        assert advance_mocked.call_count == 1


@pytest.mark.parametrize(
    "min_epochs, min_steps, current_epoch, global_step, early_stop, epoch_loop_done, raise_info_msg",
    [
        (None, None, 1, 4, True, True, False),
        (None, None, 1, 10, True, True, False),
        (4, None, 1, 4, False, False, True),
        (4, 2, 1, 4, False, False, True),
        (4, None, 1, 10, False, True, False),
        (4, 3, 1, 3, False, False, True),
        (4, 10, 1, 10, False, True, False),
        (None, 4, 1, 4, True, True, False),
    ],
)
def test_should_stop_early_stopping_conditions_not_met(
    caplog, min_epochs, min_steps, current_epoch, global_step, early_stop, epoch_loop_done, raise_info_msg
):
    """Test that checks that info message is logged when users sets `should_stop` but min conditions are not
    met."""
    trainer = Trainer(min_epochs=min_epochs, min_steps=min_steps, limit_val_batches=0)
    trainer.num_training_batches = 10
    trainer.should_stop = True
    trainer.fit_loop.epoch_loop.optimizer_loop.optim_progress.optimizer.step.total.completed = global_step
    trainer.fit_loop.epoch_loop.batch_progress.current.ready = global_step
    trainer.fit_loop.epoch_progress.current.completed = current_epoch - 1

    message = f"min_epochs={min_epochs}` or `min_steps={min_steps}` has not been met. Training will continue"
    with caplog.at_level(logging.INFO, logger="pytorch_lightning.loops"):
        assert trainer.fit_loop.epoch_loop.done is epoch_loop_done

    assert (message in caplog.text) is raise_info_msg
    assert trainer.fit_loop._should_stop_early is early_stop
