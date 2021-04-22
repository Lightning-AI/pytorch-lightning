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

import pytest
import torch

from pytorch_lightning import callbacks, seed_everything, Trainer
from tests.helpers import BoringModel
from tests.helpers.runif import RunIf


@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
def test_mc_called(tmpdir):
    seed_everything(1234)

    # -----------------
    # TRAIN LOOP ONLY
    # -----------------
    train_step_only_model = BoringModel()
    train_step_only_model.validation_step = None

    # no callback
    trainer = Trainer(max_epochs=3, checkpoint_callback=False)
    trainer.fit(train_step_only_model)
    assert len(trainer.dev_debugger.checkpoint_callback_history) == 0

    # -----------------
    # TRAIN + VAL LOOP ONLY
    # -----------------
    val_train_model = BoringModel()
    # no callback
    trainer = Trainer(max_epochs=3, checkpoint_callback=False)
    trainer.fit(val_train_model)
    assert len(trainer.dev_debugger.checkpoint_callback_history) == 0


@mock.patch('torch.save')
@pytest.mark.parametrize(
    ['epochs', 'val_check_interval', 'expected'],
    [(1, 1.0, 1), (2, 1.0, 2), (1, 0.25, 4), (2, 0.3, 7)],
)
def test_default_checkpoint_freq(save_mock, tmpdir, epochs: int, val_check_interval: float, expected: int):

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=epochs,
        weights_summary=None,
        val_check_interval=val_check_interval,
        progress_bar_refresh_rate=0,
    )
    trainer.fit(model)

    # make sure types are correct
    assert save_mock.call_count == expected


@mock.patch('torch.save')
@pytest.mark.parametrize(['k', 'epochs', 'val_check_interval', 'expected'], [
    (1, 1, 1.0, 1),
    (2, 2, 1.0, 2),
    (2, 1, 0.25, 4),
    (2, 2, 0.3, 7),
])
def test_top_k(save_mock, tmpdir, k: int, epochs: int, val_check_interval: float, expected: int):

    class TestModel(BoringModel):

        def __init__(self):
            super().__init__()
            self.last_coeff = 10.0

        def training_step(self, batch, batch_idx):
            loss = self.step(torch.ones(32))
            loss = loss / (loss + 0.0000001)
            loss += self.last_coeff
            self.log('my_loss', loss)
            self.last_coeff *= 0.999
            return loss

    model = TestModel()
    trainer = Trainer(
        callbacks=[callbacks.ModelCheckpoint(dirpath=tmpdir, monitor='my_loss', save_top_k=k)],
        default_root_dir=tmpdir,
        max_epochs=epochs,
        weights_summary=None,
        val_check_interval=val_check_interval
    )
    trainer.fit(model)

    # make sure types are correct
    assert save_mock.call_count == expected


@mock.patch('torch.save')
@RunIf(special=True, min_gpus=2)
@pytest.mark.parametrize(['k', 'epochs', 'val_check_interval', 'expected'], [(1, 1, 1.0, 1), (2, 2, 0.3, 5)])
def test_top_k_ddp(save_mock, tmpdir, k, epochs, val_check_interval, expected):

    class TestModel(BoringModel):

        def training_step(self, batch, batch_idx):
            local_rank = int(os.getenv("LOCAL_RANK"))
            self.log('my_loss', batch_idx * (1 + local_rank), on_epoch=True)
            return super().training_step(batch, batch_idx)

        def training_epoch_end(self, outputs) -> None:
            local_rank = int(os.getenv("LOCAL_RANK"))
            if self.trainer.is_global_zero:
                self.log('my_loss_2', (1 + local_rank), on_epoch=True)
            data = str(self.global_rank)
            obj = [[data], (data, ), set(data)]
            out = self.trainer.training_type_plugin.broadcast(obj)
            assert obj == [[str(self.global_rank)], (str(self.global_rank), ), set(str(self.global_rank))]
            assert out == [['0'], ('0', ), set('0')]

    model = TestModel()
    trainer = Trainer(
        callbacks=[callbacks.ModelCheckpoint(dirpath=tmpdir, monitor='my_loss_step', save_top_k=k, mode="max")],
        default_root_dir=tmpdir,
        max_epochs=epochs,
        weights_summary=None,
        val_check_interval=val_check_interval,
        accelerator="ddp",
        gpus=2,
        limit_train_batches=64,
        limit_val_batches=32,
    )
    trainer.fit(model)
    if os.getenv("LOCAL_RANK") == "0":
        assert save_mock.call_count == expected
