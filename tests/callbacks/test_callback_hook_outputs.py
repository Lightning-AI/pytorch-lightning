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

from pytorch_lightning import Callback, Trainer
from tests.helpers.boring_model import BoringModel


@pytest.mark.parametrize("single_cb", [False, True])
def test_train_step_no_return(tmpdir, single_cb: bool):
    """
    Tests that only training_step can be used
    """

    class CB(Callback):

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
            assert 'loss' in outputs

        def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
            assert 'x' in outputs

        def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
            assert 'x' in outputs

    class TestModel(BoringModel):

        def on_train_batch_end(self, outputs, batch, batch_idx: int, dataloader_idx: int) -> None:
            assert 'loss' in outputs

        def on_validation_batch_end(self, outputs, batch, batch_idx: int, dataloader_idx: int) -> None:
            assert 'x' in outputs

        def on_test_batch_end(self, outputs, batch, batch_idx: int, dataloader_idx: int) -> None:
            assert 'x' in outputs

        def training_epoch_end(self, outputs) -> None:
            assert len(outputs) == self.trainer.num_training_batches

    model = TestModel()

    trainer = Trainer(
        callbacks=CB() if single_cb else [CB()],
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
        weights_summary=None,
    )

    assert any(isinstance(c, CB) for c in trainer.callbacks)

    trainer.fit(model)


def test_free_memory_on_eval_outputs(tmpdir):

    class CB(Callback):

        def on_epoch_end(self, trainer, pl_module):
            assert len(trainer.evaluation_loop.outputs) == 0

    model = BoringModel()

    trainer = Trainer(
        callbacks=CB(),
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=1,
        weights_summary=None,
    )

    trainer.fit(model)
