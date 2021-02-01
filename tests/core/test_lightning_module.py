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
from unittest.mock import Mock, patch

import pytest
from torch.optim import Adam, SGD

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.base import BoringModel


def test_property_current_epoch():
    """ Test that the current_epoch in LightningModule is accessible via the Trainer. """
    model = BoringModel()
    assert model.current_epoch == 0

    trainer = Mock(current_epoch=123)
    model.trainer = trainer
    assert model.current_epoch == 123


def test_property_global_step():
    """ Test that the global_step in LightningModule is accessible via the Trainer. """
    model = BoringModel()
    assert model.global_step == 0

    trainer = Mock(global_step=123)
    model.trainer = trainer
    assert model.global_step == 123


def test_property_global_rank():
    """ Test that the global rank in LightningModule is accessible via the Trainer. """
    model = BoringModel()
    assert model.global_rank == 0

    trainer = Mock(global_rank=123)
    model.trainer = trainer
    assert model.global_rank == 123


def test_property_local_rank():
    """ Test that the local rank in LightningModule is accessible via the Trainer. """
    model = BoringModel()
    assert model.local_rank == 0

    trainer = Mock(local_rank=123)
    model.trainer = trainer
    assert model.local_rank == 123


def test_property_logger(tmpdir):
    """ Test that the logger in LightningModule is accessible via the Trainer. """
    model = BoringModel()
    assert model.logger is None

    logger = TensorBoardLogger(tmpdir)
    trainer = Mock(logger=logger)
    model.trainer = trainer
    assert model.logger == logger


def test_automatic_optimization(tmpdir):
    class TestModel(BoringModel):
        def optimizer_step(self, *_, **__):
            pass

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        accumulate_grad_batches=2,
    )

    with pytest.raises(
        MisconfigurationException,
        match='overriding .* optimizer_step .* `accumulate_grad_batches` .* should be 1'
    ):
        trainer.fit(model)


def test_automatic_optimization_num_calls(tmpdir):

    with patch("torch.optim.SGD.step") as sgd_step, \
         patch("torch.optim.SGD.zero_grad") as sgd_zero_grad, \
         patch("torch.optim.Adam.step") as adam_step, \
         patch("torch.optim.Adam.zero_grad") as adam_zero_grad:

        class TestModel(BoringModel):

            def training_step(self, batch, batch_idx, optimizer_idx):
                output = self.layer(batch)
                loss = self.loss(batch, output)
                return {"loss": loss}

            def configure_optimizers(self):
                optimizer = SGD(self.layer.parameters(), lr=0.1)
                optimizer_2 = Adam(self.layer.parameters(), lr=0.1)
                return [optimizer, optimizer_2]

            def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                               optimizer_closure, on_tpu, using_native_amp, using_lbfgs):

                assert optimizer_closure.__name__ == "train_step_and_backward_closure"

                # update generator opt every 2 steps
                if optimizer_idx == 0:
                    if batch_idx % 2 == 0:
                        assert isinstance(optimizer, SGD)
                        optimizer.step(closure=optimizer_closure)

                # update discriminator opt every 4 steps
                if optimizer_idx == 1:
                    if batch_idx % 4 == 0:
                        assert isinstance(optimizer, Adam)
                        optimizer.step(closure=optimizer_closure)

        model = TestModel()
        model.training_epoch_end = None

        trainer = Trainer(
            max_epochs=1,
            default_root_dir=tmpdir,
            limit_train_batches=8,
            limit_val_batches=1,
            accumulate_grad_batches=1,
        )

        trainer.fit(model)

    assert sgd_step.call_count == 4
    assert sgd_zero_grad.call_count == 4
    assert adam_step.call_count == 2
    assert adam_zero_grad.call_count == 2


def test_params_groups_and_state_are_accessible(tmpdir):

    class TestModel(BoringModel):

        def training_step(self, batch, batch_idx, optimizer_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            return {"loss": loss}

        def configure_optimizers(self):
            optimizer = SGD(self.layer.parameters(), lr=0.1)
            optimizer_2 = Adam(self.layer.parameters(), lr=0.1)
            return [optimizer, optimizer_2]

        def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, closure,
                           on_tpu=False, using_native_amp=False, using_lbfgs=False):
            # warm up lr
            if self.trainer.global_step < 500:
                lr_scale = min(1., float(self.trainer.global_step + 1) / 500.)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr_scale * 0.01

            optimizer.step(closure=closure)

    model = TestModel()
    model.training_epoch_end = None

    trainer = Trainer(
        max_epochs=1,
        default_root_dir=tmpdir,
        limit_train_batches=8,
        limit_val_batches=1,
        accumulate_grad_batches=1,
    )

    trainer.fit(model)
