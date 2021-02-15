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
import sys
from unittest import mock

import pytest
import torch

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

from pytorch_lightning import LightningModule  # noqa: E402
from pytorch_lightning import Trainer  # noqa: E402
from tests.helpers.boring_model import BoringModel  # noqa: E402


@pytest.mark.skipif(
    not os.getenv("PL_RUNNING_SPECIAL_TESTS", '0') == '1', reason="test should be run outside of pytest"
)
def test_logging_sync_dist_true_ddp(tmpdir):
    """
    Tests to ensure that the sync_dist flag works with CPU (should just return the original value)
    """
    fake_result = 1

    class TestModel(BoringModel):

        def training_step(self, batch, batch_idx):
            acc = self.step(batch[0])
            self.log('foo', torch.tensor(fake_result), on_step=False, on_epoch=True)
            return acc

        def validation_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            self.log('bar', torch.tensor(fake_result), on_step=False, on_epoch=True)
            return {"x": loss}

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=1,
        limit_val_batches=1,
        max_epochs=2,
        weights_summary=None,
        accelerator="ddp",
        gpus=1,
        num_nodes=2,
    )
    trainer.fit(model)

    assert trainer.logged_metrics['foo'] == fake_result
    assert trainer.logged_metrics['bar'] == fake_result


@pytest.mark.skipif(
    not os.getenv("PL_RUNNING_SPECIAL_TESTS", '0') == '1', reason="test should be run outside of pytest"
)
@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
def test__validation_step__log(tmpdir):
    """
    Tests that validation_step can log
    """

    class TestModel(BoringModel):

        def training_step(self, batch, batch_idx):
            acc = self.step(batch)
            acc = acc + batch_idx
            self.log('a', acc, on_step=True, on_epoch=True)
            self.log('a2', 2)

            self.training_step_called = True
            return acc

        def validation_step(self, batch, batch_idx):
            acc = self.step(batch)
            acc = acc + batch_idx
            self.log('b', acc, on_step=True, on_epoch=True)
            self.training_step_called = True

        def backward(self, loss, optimizer, optimizer_idx):
            return LightningModule.backward(self, loss, optimizer, optimizer_idx)

    model = TestModel()
    model.validation_step_end = None
    model.validation_epoch_end = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=2,
        log_every_n_steps=1,
        weights_summary=None,
        accelerator="ddp",
        gpus=1,
        num_nodes=2,
    )
    trainer.fit(model)

    # make sure all the metrics are available for callbacks
    expected_logged_metrics = {
        'a2',
        'a_step',
        'a_epoch',
        'b_step/epoch_0',
        'b_step/epoch_1',
        'b_epoch',
        'epoch',
    }
    logged_metrics = set(trainer.logged_metrics.keys())
    assert expected_logged_metrics == logged_metrics

    # we don't want to enable val metrics during steps because it is not something that users should do
    # on purpose DO NOT allow step_b... it's silly to monitor val step metrics
    callback_metrics = set(trainer.callback_metrics.keys())
    callback_metrics.remove('debug_epoch')
    expected_cb_metrics = {'a', 'a2', 'b', 'a_epoch', 'b_epoch', 'a_step'}
    assert expected_cb_metrics == callback_metrics
