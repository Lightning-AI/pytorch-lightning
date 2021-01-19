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
Tests to ensure that the training loop works with a dict (1.0)
"""
import os
from unittest import mock

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
from tests.base.deterministic_model import DeterministicModel


@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
def test__training_step__flow_dict(tmpdir):
    """
    Tests that only training_step can be used
    """

    class TestModel(DeterministicModel):
        def training_step(self, batch, batch_idx):
            acc = self.step(batch, batch_idx)
            acc = acc + batch_idx
            self.training_step_called = True
            return {'loss': acc, 'random_things': [1, 'a', torch.tensor(2)]}

        def backward(self, loss, optimizer, optimizer_idx):
            return LightningModule.backward(self, loss, optimizer, optimizer_idx)

    model = TestModel()
    model.val_dataloader = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=2,
        log_every_n_steps=1,
        weights_summary=None,
    )
    trainer.fit(model)

    # make sure correct steps were called
    assert model.training_step_called
    assert not model.training_step_end_called
    assert not model.training_epoch_end_called


@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
def test__training_step__tr_step_end__flow_dict(tmpdir):
    """
    Tests that only training_step can be used
    """

    class TestModel(DeterministicModel):
        def training_step(self, batch, batch_idx):
            acc = self.step(batch, batch_idx)
            acc = acc + batch_idx
            self.training_step_called = True
            self.out = {'loss': acc, 'random_things': [1, 'a', torch.tensor(2)]}
            return self.out

        def training_step_end(self, tr_step_output):
            assert tr_step_output == self.out
            assert self.count_num_graphs(tr_step_output) == 1
            self.training_step_end_called = True
            return tr_step_output

        def backward(self, loss, optimizer, optimizer_idx):
            return LightningModule.backward(self, loss, optimizer, optimizer_idx)

    model = TestModel()
    model.val_dataloader = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=2,
        log_every_n_steps=1,
        weights_summary=None,
    )
    trainer.fit(model)

    # make sure correct steps were called
    assert model.training_step_called
    assert model.training_step_end_called
    assert not model.training_epoch_end_called


@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
def test__training_step__epoch_end__flow_dict(tmpdir):
    """
    Tests that only training_step can be used
    """

    class TestModel(DeterministicModel):
        def training_step(self, batch, batch_idx):
            acc = self.step(batch, batch_idx)
            acc = acc + batch_idx

            self.training_step_called = True
            out = {'loss': acc, 'random_things': [1, 'a', torch.tensor(2)]}
            return out

        def training_epoch_end(self, outputs):
            self.training_epoch_end_called = True

            # verify we saw the current num of batches
            assert len(outputs) == 2

            for b in outputs:
                assert isinstance(b, dict)
                assert self.count_num_graphs(b) == 0
                assert {'random_things', 'loss'} == set(b.keys())

        def backward(self, loss, optimizer, optimizer_idx):
            return LightningModule.backward(self, loss, optimizer, optimizer_idx)

    model = TestModel()
    model.val_dataloader = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=2,
        log_every_n_steps=1,
        weights_summary=None,
    )
    trainer.fit(model)

    # make sure correct steps were called
    assert model.training_step_called
    assert not model.training_step_end_called
    assert model.training_epoch_end_called


@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
def test__training_step__step_end__epoch_end__flow_dict(tmpdir):
    """
    Tests that only training_step can be used
    """

    class TestModel(DeterministicModel):
        def training_step(self, batch, batch_idx):
            acc = self.step(batch, batch_idx)
            acc = acc + batch_idx

            self.training_step_called = True
            self.out = {'loss': acc, 'random_things': [1, 'a', torch.tensor(2)]}
            return self.out

        def training_step_end(self, tr_step_output):
            assert tr_step_output == self.out
            assert self.count_num_graphs(tr_step_output) == 1
            self.training_step_end_called = True
            return tr_step_output

        def training_epoch_end(self, outputs):
            self.training_epoch_end_called = True

            # verify we saw the current num of batches
            assert len(outputs) == 2

            for b in outputs:
                assert isinstance(b, dict)
                assert self.count_num_graphs(b) == 0
                assert {'random_things', 'loss'} == set(b.keys())

        def backward(self, loss, optimizer, optimizer_idx):
            return LightningModule.backward(self, loss, optimizer, optimizer_idx)

    model = TestModel()
    model.val_dataloader = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=2,
        log_every_n_steps=1,
        weights_summary=None,
    )
    trainer.fit(model)

    # make sure correct steps were called
    assert model.training_step_called
    assert model.training_step_end_called
    assert model.training_epoch_end_called
