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
import platform
from unittest import mock

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.utilities import _TORCH_GREATER_EQUAL_1_6
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers import BoringModel, RandomDataset

if _TORCH_GREATER_EQUAL_1_6:
    from pytorch_lightning.callbacks import StochasticWeightAveraging

    class SwaTestModel(BoringModel):

        def __init__(self, batchnorm: bool = True):
            super().__init__()
            layers = [nn.Linear(32, 32)]
            if batchnorm:
                layers.append(nn.BatchNorm1d(32))
            layers += [nn.ReLU(), nn.Linear(32, 2)]
            self.layer = nn.Sequential(*layers)

        def training_step(self, batch, batch_idx):
            output = self.forward(batch)
            loss = self.loss(batch, output)
            return {"loss": loss}

        def train_dataloader(self):
            return DataLoader(RandomDataset(32, 64), batch_size=2)

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
            assert trainer.train_loop._skip_backward == (trainer.current_epoch > self.swa_end)

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
            assert not trainer.train_loop._skip_backward
            assert trainer.accumulate_grad_batches == 2
            assert trainer.num_training_batches == 5

            # check backward call count. the batchnorm update epoch should not backward
            assert trainer.dev_debugger.count_events(
                "backward_call"
            ) == trainer.max_epochs * trainer.limit_train_batches

            # check call counts
            assert self.update_parameters_calls == trainer.max_epochs - (self._swa_epoch_start - 1)
            assert self.transfer_weights_calls == 1


@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
def train_with_swa(tmpdir, batchnorm=True, accelerator=None, gpus=None, num_processes=1):
    model = SwaTestModel(batchnorm=batchnorm)
    swa_start = 2
    max_epochs = 5
    swa_callback = SwaTestCallback(swa_epoch_start=swa_start, swa_lrs=0.1)
    assert swa_callback.update_parameters_calls == 0
    assert swa_callback.transfer_weights_calls == 0

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=max_epochs,
        limit_train_batches=5,
        limit_val_batches=0,
        callbacks=[swa_callback],
        accumulate_grad_batches=2,
        accelerator=accelerator,
        gpus=gpus,
        num_processes=num_processes
    )
    trainer.fit(model)

    # check the model is the expected
    assert trainer.lightning_module == model


@pytest.mark.skipif(not _TORCH_GREATER_EQUAL_1_6, reason="SWA available from PyTorch 1.6.0")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.skipif(
    not os.getenv("PL_RUNNING_SPECIAL_TESTS", '0') == '1', reason="test should be run outside of pytest"
)
def test_swa_callback_ddp(tmpdir):
    train_with_swa(tmpdir, accelerator="ddp", gpus=2)


@pytest.mark.skipif(not _TORCH_GREATER_EQUAL_1_6, reason="SWA available from PyTorch 1.6.0")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_swa_callback_ddp_spawn(tmpdir):
    train_with_swa(tmpdir, accelerator="ddp_spawn", gpus=2)


@pytest.mark.skipif(not _TORCH_GREATER_EQUAL_1_6, reason="SWA available from PyTorch 1.6.0")
@pytest.mark.skipif(platform.system() == "Windows", reason="ddp_cpu is not available on Windows")
def test_swa_callback_ddp_cpu(tmpdir):
    train_with_swa(tmpdir, accelerator="ddp_cpu", num_processes=2)


@pytest.mark.skipif(not _TORCH_GREATER_EQUAL_1_6, reason="SWA available from PyTorch 1.6.0")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires a GPU machine")
def test_swa_callback_1_gpu(tmpdir):
    train_with_swa(tmpdir, gpus=1)


@pytest.mark.skipif(not _TORCH_GREATER_EQUAL_1_6, reason="SWA available from PyTorch 1.6.0")
@pytest.mark.parametrize("batchnorm", (True, False))
def test_swa_callback(tmpdir, batchnorm):
    train_with_swa(tmpdir, batchnorm=batchnorm)


@pytest.mark.skipif(not _TORCH_GREATER_EQUAL_1_6, reason="SWA available from PyTorch 1.6.0")
def test_swa_raises():
    with pytest.raises(MisconfigurationException, match=">0 integer or a float between 0 and 1"):
        StochasticWeightAveraging(swa_epoch_start=0, swa_lrs=0.1)
    with pytest.raises(MisconfigurationException, match=">0 integer or a float between 0 and 1"):
        StochasticWeightAveraging(swa_epoch_start=1.5, swa_lrs=0.1)
    with pytest.raises(MisconfigurationException, match=">0 integer or a float between 0 and 1"):
        StochasticWeightAveraging(swa_epoch_start=-1, swa_lrs=0.1)
    with pytest.raises(MisconfigurationException, match="positive float or a list of positive float"):
        StochasticWeightAveraging(swa_epoch_start=5, swa_lrs=[0.2, 1])


@pytest.mark.parametrize('stochastic_weight_avg', [False, True])
@pytest.mark.parametrize('use_callbacks', [False, True])
@pytest.mark.skipif(not _TORCH_GREATER_EQUAL_1_6, reason="SWA available from PyTorch 1.6.0")
def test_trainer_and_stochastic_weight_avg(tmpdir, use_callbacks, stochastic_weight_avg):
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
        assert len([cb for cb in trainer.callbacks if isinstance(cb, StochasticWeightAveraging)]) == 1
        assert trainer.callbacks[0]._swa_lrs == (1e-3 if use_callbacks else 0.1)
    else:
        assert all(not isinstance(cb, StochasticWeightAveraging) for cb in trainer.callbacks)
