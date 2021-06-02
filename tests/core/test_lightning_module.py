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
from torch import nn
from torch.optim import Adam, SGD

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers import BoringModel
from tests.helpers.runif import RunIf


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


def test_automatic_optimization_raises(tmpdir):

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
        MisconfigurationException, match='overriding .* optimizer_step .* `accumulate_grad_batches` .* should be 1'
    ):
        trainer.fit(model)


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

        def optimizer_step(
            self,
            epoch,
            batch_idx,
            optimizer,
            optimizer_idx,
            optimizer_closure,
            on_tpu=False,
            using_native_amp=False,
            using_lbfgs=False
        ):
            # warm up lr
            if self.trainer.global_step < 500:
                lr_scale = min(1., float(self.trainer.global_step + 1) / 500.)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr_scale * 0.01

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


def test_toggle_untoggle_2_optimizers_no_shared_parameters(tmpdir):

    class TestModel(BoringModel):

        def training_step(self, batch, batch_idx, optimizer_idx=None):
            return super().training_step(batch, batch_idx)

        def __init__(self):
            super().__init__()
            self.layer_1 = nn.Sequential(
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
            )

            self.layer_2 = nn.Sequential(
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 2),
            )

            # set some weights to False to check untoggle works as expected.
            self.layer_1[2].weight.requires_grad = False
            self.layer_1[4].weight.requires_grad = False

            self.layer_2[1].weight.requires_grad = False
            self.layer_2[3].weight.requires_grad = False

        def configure_optimizers(self):
            optimizer = SGD(self.layer_1.parameters(), lr=0.1)
            optimizer_2 = Adam(self.layer_2.parameters(), lr=0.1)
            return [optimizer, optimizer_2]

        def optimizer_step(
            self,
            current_epoch,
            batch_nb,
            optimizer,
            optimizer_idx,
            closure,
            on_tpu=False,
            using_native_amp=False,
            using_lbfgs=False
        ):
            if optimizer_idx == 0:
                assert self.layer_1[0].weight.requires_grad is True
                assert self.layer_1[2].weight.requires_grad is False
                assert self.layer_1[4].weight.requires_grad is False

                assert self.layer_2[1].weight.requires_grad is False
                assert self.layer_2[3].weight.requires_grad is False
                assert self.layer_2[5].weight.requires_grad is False

            if optimizer_idx == 1:
                assert self.layer_1[0].weight.requires_grad is False
                assert self.layer_1[2].weight.requires_grad is False
                assert self.layer_1[4].weight.requires_grad is False

                assert self.layer_2[1].weight.requires_grad is False
                assert self.layer_2[3].weight.requires_grad is False
                assert self.layer_2[5].weight.requires_grad is True

            optimizer.step(closure=closure)

    model = TestModel()
    model.training_epoch_end = None

    trainer = Trainer(
        max_epochs=1,
        default_root_dir=tmpdir,
        limit_train_batches=8,
        accumulate_grad_batches=1,
        limit_val_batches=0,
    )
    trainer.fit(model)


def test_toggle_untoggle_3_optimizers_shared_parameters(tmpdir):

    class TestModel(BoringModel):

        def __init__(self):
            super().__init__()
            self.layer_1 = nn.Sequential(
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
            )

            self.layer_2 = nn.Sequential(
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 2),
            )

            self.layer_3 = nn.Sequential(
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 2),
            )

            # set some weights to False to check untoggle works as expected.
            self.layer_1[2].weight.requires_grad = False
            self.layer_1[4].weight.requires_grad = False

            self.layer_2[1].weight.requires_grad = False
            self.layer_2[3].weight.requires_grad = False

            self.layer_3[1].weight.requires_grad = False
            self.layer_3[5].weight.requires_grad = False

        def optimizer_step(
            self,
            current_epoch,
            batch_nb,
            optimizer,
            optimizer_idx,
            closure,
            on_tpu=False,
            using_native_amp=False,
            using_lbfgs=False
        ):
            if optimizer_idx == 0:
                assert self.layer_1[0].weight.requires_grad is True
                assert self.layer_1[2].weight.requires_grad is False
                assert self.layer_1[4].weight.requires_grad is False

                assert self.layer_2[1].weight.requires_grad is False
                assert self.layer_2[3].weight.requires_grad is False
                assert self.layer_2[5].weight.requires_grad is True

                assert self.layer_3[1].weight.requires_grad is False
                assert self.layer_3[3].weight.requires_grad is False
                assert self.layer_3[5].weight.requires_grad is False

            if optimizer_idx == 1:
                assert self.layer_1[0].weight.requires_grad is False
                assert self.layer_1[2].weight.requires_grad is False
                assert self.layer_1[4].weight.requires_grad is False

                assert self.layer_2[1].weight.requires_grad is False
                assert self.layer_2[3].weight.requires_grad is False
                assert self.layer_2[5].weight.requires_grad is True

                assert self.layer_3[1].weight.requires_grad is False
                assert self.layer_3[3].weight.requires_grad is True
                assert self.layer_3[5].weight.requires_grad is False

            if optimizer_idx == 2:
                assert self.layer_1[0].weight.requires_grad is True
                assert self.layer_1[2].weight.requires_grad is False
                assert self.layer_1[4].weight.requires_grad is False

                assert self.layer_2[1].weight.requires_grad is False
                assert self.layer_2[3].weight.requires_grad is False
                assert self.layer_2[5].weight.requires_grad is False

                assert self.layer_3[1].weight.requires_grad is False
                assert self.layer_3[3].weight.requires_grad is True
                assert self.layer_3[5].weight.requires_grad is False

            optimizer.step(closure=closure)

        def training_step(self, batch, batch_idx, optimizer_idx=None):
            loss = super().training_step(batch, batch_idx)
            # make sure the model is untoggle when returning None
            return loss if batch_idx % 2 == 0 else None

        @staticmethod
        def combine_generators(gen_1, gen_2):
            for p in gen_1:
                yield p
            for p in gen_2:
                yield p

        def configure_optimizers(self):
            optimizer_1 = SGD(self.combine_generators(
                self.layer_1.parameters(),
                self.layer_2.parameters(),
            ), lr=0.1)
            optimizer_2 = Adam(self.combine_generators(
                self.layer_2.parameters(),
                self.layer_3.parameters(),
            ), lr=0.1)
            optimizer_3 = SGD(self.combine_generators(
                self.layer_3.parameters(),
                self.layer_1.parameters(),
            ), lr=0.1)
            return [optimizer_1, optimizer_2, optimizer_3]

    model = TestModel()
    model.training_epoch_end = None

    trainer = Trainer(
        max_epochs=1,
        default_root_dir=tmpdir,
        limit_train_batches=8,
        accumulate_grad_batches=1,
    )

    trainer.fit(model)


@RunIf(min_gpus=1)
def test_device_placement(tmpdir):

    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True, gpus=1)
    trainer.fit(model)

    def assert_device(device: torch.device) -> None:
        assert model.device == device
        for p in model.parameters():
            assert p.device == device

    assert_device(torch.device("cpu"))
    model.to(torch.device("cuda:0"))
    assert_device(torch.device("cuda:0"))
    trainer.test(model)
    assert_device(torch.device("cpu"))
    trainer.predict(model, dataloaders=model.train_dataloader())
    assert_device(torch.device("cpu"))
