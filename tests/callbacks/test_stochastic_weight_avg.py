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
from unittest import mock

import pytest
import torch
from torch import nn
from torch.optim.swa_utils import SWALR
from torch.utils.data import DataLoader

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.callbacks import StochasticWeightAveraging
from pytorch_lightning.plugins import DDPSpawnPlugin
from pytorch_lightning.trainer.connectors.data_connector import _PatchDataLoader
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers.boring_model import BoringModel, RandomDataset, RandomIterableDataset
from tests.helpers.runif import RunIf


class SwaTestModel(BoringModel):
    def __init__(self, batchnorm: bool = True, interval: str = "epoch", iterable_dataset: bool = False):
        super().__init__()
        layers = [nn.Linear(32, 32)]
        if batchnorm:
            layers.append(nn.BatchNorm1d(32))
        layers += [nn.ReLU(), nn.Linear(32, 2)]
        self.layer = nn.Sequential(*layers)
        self.interval = interval
        self.iterable_dataset = iterable_dataset

    def training_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = self.loss(batch, output)
        return {"loss": loss}

    def train_dataloader(self):

        dset_cls = RandomIterableDataset if self.iterable_dataset else RandomDataset
        dset = dset_cls(32, 64)

        return DataLoader(dset, batch_size=2)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=1),
                "interval": self.interval,
            },
        }


class SwaTestCallback(StochasticWeightAveraging):
    update_parameters_calls: int = 0
    transfer_weights_calls: int = 0

    def update_parameters(self, *args, **kwargs):
        self.update_parameters_calls += 1
        return StochasticWeightAveraging.update_parameters(*args, **kwargs)

    def transfer_weights(self, *args, **kwargs):
        self.transfer_weights_calls += 1
        return StochasticWeightAveraging.transfer_weights(*args, **kwargs)

    def on_train_epoch_start(self, trainer, *args):
        super().on_train_epoch_start(trainer, *args)
        assert trainer.fit_loop._skip_backward == (trainer.current_epoch > self.swa_end)
        if self.swa_start <= trainer.current_epoch:
            assert isinstance(trainer.lr_schedulers[0]["scheduler"], SWALR)
            assert trainer.lr_schedulers[0]["interval"] == "epoch"
            assert trainer.lr_schedulers[0]["frequency"] == 1

    def on_train_epoch_end(self, trainer, *args):
        super().on_train_epoch_end(trainer, *args)
        if self.swa_start <= trainer.current_epoch <= self.swa_end:
            swa_epoch = trainer.current_epoch - self.swa_start
            assert self.n_averaged == swa_epoch + 1
        elif trainer.current_epoch > self.swa_end:
            assert self.n_averaged == self._max_epochs - self.swa_start

    def on_train_end(self, trainer, pl_module):
        super().on_train_end(trainer, pl_module)

        # make sure these are correctly set again
        assert not trainer.fit_loop._skip_backward
        assert trainer.accumulate_grad_batches == 2
        assert trainer.num_training_batches == 5

        if not isinstance(trainer.training_type_plugin, DDPSpawnPlugin):
            # check backward call count. the batchnorm update epoch should not backward
            assert trainer.accelerator.backward.call_count == trainer.max_epochs * trainer.limit_train_batches

        # check call counts
        assert self.update_parameters_calls == trainer.max_epochs - (self._swa_epoch_start - 1)
        assert self.transfer_weights_calls == 1


def train_with_swa(
    tmpdir, batchnorm=True, accelerator=None, gpus=None, num_processes=1, interval="epoch", iterable_dataset=False
):
    model = SwaTestModel(batchnorm=batchnorm, interval=interval, iterable_dataset=iterable_dataset)
    swa_start = 2
    max_epochs = 5
    swa_callback = SwaTestCallback(swa_epoch_start=swa_start, swa_lrs=0.1)
    assert swa_callback.update_parameters_calls == 0
    assert swa_callback.transfer_weights_calls == 0

    trainer = Trainer(
        default_root_dir=tmpdir,
        progress_bar_refresh_rate=0,
        max_epochs=max_epochs,
        limit_train_batches=5,
        limit_val_batches=0,
        callbacks=[swa_callback],
        accumulate_grad_batches=2,
        accelerator=accelerator,
        gpus=gpus,
        num_processes=num_processes,
    )

    with mock.patch.object(Accelerator, "backward", wraps=trainer.accelerator.backward):
        trainer.fit(model)

    # check the model is the expected
    assert trainer.lightning_module == model


@RunIf(min_gpus=2, special=True)
def test_swa_callback_ddp(tmpdir):
    train_with_swa(tmpdir, accelerator="ddp", gpus=2)


@RunIf(min_gpus=2)
def test_swa_callback_ddp_spawn(tmpdir):
    train_with_swa(tmpdir, accelerator="ddp_spawn", gpus=2)


@RunIf(skip_windows=True)
def test_swa_callback_ddp_cpu(tmpdir):
    train_with_swa(tmpdir, accelerator="ddp_cpu", num_processes=2)


@RunIf(min_gpus=1)
def test_swa_callback_1_gpu(tmpdir):
    train_with_swa(tmpdir, gpus=1)


@pytest.mark.parametrize("batchnorm", (True, False))
@pytest.mark.parametrize("iterable_dataset", (True, False))
def test_swa_callback(tmpdir, batchnorm: bool, iterable_dataset: bool):
    train_with_swa(tmpdir, batchnorm=batchnorm, iterable_dataset=iterable_dataset)


@pytest.mark.parametrize("interval", ("epoch", "step"))
def test_swa_callback_scheduler_step(tmpdir, interval: str):
    train_with_swa(tmpdir, interval=interval)


def test_swa_warns(tmpdir, caplog):
    model = SwaTestModel(interval="step")
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True, stochastic_weight_avg=True)
    with caplog.at_level(level=logging.INFO), pytest.warns(UserWarning, match="SWA is currently only supported"):
        trainer.fit(model)
    assert "Swapping scheduler" in caplog.text


def test_swa_raises():
    with pytest.raises(MisconfigurationException, match=">0 integer or a float between 0 and 1"):
        StochasticWeightAveraging(swa_epoch_start=0, swa_lrs=0.1)
    with pytest.raises(MisconfigurationException, match=">0 integer or a float between 0 and 1"):
        StochasticWeightAveraging(swa_epoch_start=1.5, swa_lrs=0.1)
    with pytest.raises(MisconfigurationException, match=">0 integer or a float between 0 and 1"):
        StochasticWeightAveraging(swa_epoch_start=-1, swa_lrs=0.1)
    with pytest.raises(MisconfigurationException, match="positive float or a list of positive float"):
        StochasticWeightAveraging(swa_epoch_start=5, swa_lrs=[0.2, 1])


@pytest.mark.parametrize("stochastic_weight_avg", [False, True])
@pytest.mark.parametrize("use_callbacks", [False, True])
def test_trainer_and_stochastic_weight_avg(tmpdir, use_callbacks: bool, stochastic_weight_avg: bool):
    """Test to ensure SWA Callback is injected when `stochastic_weight_avg` is provided to the Trainer"""

    class TestModel(BoringModel):
        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            return optimizer

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=StochasticWeightAveraging(swa_lrs=1e-3) if use_callbacks else None,
        stochastic_weight_avg=stochastic_weight_avg,
        limit_train_batches=4,
        limit_val_batches=4,
        max_epochs=2,
    )
    trainer.fit(model)
    if use_callbacks or stochastic_weight_avg:
        assert sum(1 for cb in trainer.callbacks if isinstance(cb, StochasticWeightAveraging)) == 1
        assert trainer.callbacks[0]._swa_lrs == [1e-3 if use_callbacks else 0.1]
    else:
        assert all(not isinstance(cb, StochasticWeightAveraging) for cb in trainer.callbacks)


def test_swa_deepcopy(tmpdir):
    """Test to ensure SWA Callback doesn't deepcopy dataloaders and datamodule potentially leading to OOM"""

    class TestSWA(StochasticWeightAveraging):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.on_before_accelerator_backend_setup_called = False

        def on_before_accelerator_backend_setup(self, trainer: "Trainer", pl_module: "LightningModule"):
            super().on_before_accelerator_backend_setup(trainer, pl_module)
            assert self._average_model.train_dataloader is not pl_module.train_dataloader
            assert self._average_model.train_dataloader.__self__ == self._average_model
            assert isinstance(pl_module.train_dataloader, _PatchDataLoader)
            assert self._average_model.trainer is None
            self.on_before_accelerator_backend_setup_called = True

    model = BoringModel()
    swa = TestSWA()
    trainer = Trainer(default_root_dir=tmpdir, callbacks=swa, fast_dev_run=True)
    trainer.fit(model, train_dataloader=DataLoader(RandomDataset(32, 2)))
    assert swa.on_before_accelerator_backend_setup_called


def test_swa_multiple_lrs(tmpdir):
    swa_lrs = [0.123, 0.321]

    class TestModel(BoringModel):
        def __init__(self):
            super(BoringModel, self).__init__()
            self.layer1 = torch.nn.Linear(32, 32)
            self.layer2 = torch.nn.Linear(32, 2)

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            return x

        def configure_optimizers(self):
            params = [{"params": self.layer1.parameters(), "lr": 0.1}, {"params": self.layer2.parameters(), "lr": 0.2}]
            return torch.optim.Adam(params)

        def on_train_epoch_start(self):
            optimizer = trainer.optimizers[0]
            assert [pg["lr"] for pg in optimizer.param_groups] == [0.1, 0.2]
            assert [pg["initial_lr"] for pg in optimizer.param_groups] == swa_lrs
            assert [pg["swa_lr"] for pg in optimizer.param_groups] == swa_lrs
            self.on_train_epoch_start_called = True

    model = TestModel()
    swa_callback = StochasticWeightAveraging(swa_lrs=swa_lrs)
    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=swa_callback,
        fast_dev_run=1,
    )
    trainer.fit(model)
    assert model.on_train_epoch_start_called
