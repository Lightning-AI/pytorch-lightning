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
import torch
import pytest
from tests.base.boring_model import BoringModel, RandomDataset
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import APEX_AVAILABLE
from pytorch_lightning.utilities.exceptions import MisconfigurationException


def test_logdir(tmpdir):
    """
    Tests that the path is correct when checkpoint and loggers are used
    """
    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)

            expected = os.path.join(self.trainer.default_root_dir, 'lightning_logs', 'version_0')
            assert self.trainer.log_dir == expected
            return {"loss": loss}

    model = TestModel()

    limit_train_batches = 2
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=limit_train_batches,
        limit_val_batches=2,
        max_epochs=1,
    )

    trainer.fit(model)


def test_logdir_no_checkpoint_cb(tmpdir):
    """
    Tests that the path is correct with no checkpoint
    """
    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            expected = os.path.join(self.trainer.default_root_dir, 'lightning_logs', 'version_0')
            assert self.trainer.log_dir == expected
            return {"loss": loss}

    model = TestModel()

    limit_train_batches = 2
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=limit_train_batches,
        limit_val_batches=2,
        max_epochs=1,
        checkpoint_callback=False
    )

    trainer.fit(model)


def test_logdir_no_logger(tmpdir):
    """
    Tests that the path is correct even when there is no logger
    """
    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            expected = os.path.join(self.trainer.default_root_dir)
            assert self.trainer.log_dir == expected
            return {"loss": loss}

    model = TestModel()

    limit_train_batches = 2
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=limit_train_batches,
        limit_val_batches=2,
        max_epochs=1,
        logger=False,
    )

    trainer.fit(model)


def test_logdir_no_logger_no_checkpoint(tmpdir):
    """
    Tests that the path is correct even when there is no logger
    """
    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            expected = os.path.join(self.trainer.default_root_dir)
            assert self.trainer.log_dir == expected
            return {"loss": loss}

    model = TestModel()

    limit_train_batches = 2
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=limit_train_batches,
        limit_val_batches=2,
        max_epochs=1,
        logger=False,
        checkpoint_callback=False
    )

    trainer.fit(model)
