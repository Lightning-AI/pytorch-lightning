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
import torch
import torch.nn as nn

from pytorch_lightning import Callback, Trainer
from pytorch_lightning.utilities.device_dtype_mixin import DeviceDtypeModuleMixin
from tests.helpers import BoringModel
from tests.helpers.runif import RunIf


class SubSubModule(DeviceDtypeModuleMixin):
    pass


class SubModule(nn.Module):

    def __init__(self):
        super().__init__()
        self.module = SubSubModule()


class TopModule(BoringModel):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.module = SubModule()


class DeviceAssertCallback(Callback):

    def on_train_batch_start(self, trainer, model, batch, batch_idx, dataloader_idx):
        rank = trainer.local_rank
        assert isinstance(model, TopModule)
        # index = None also means first device
        assert (model.device.index is None and rank == 0) or model.device.index == rank
        assert model.device == model.module.module.device


@pytest.mark.parametrize(['dst_dtype'], [
    pytest.param(torch.float),
    pytest.param(torch.double),
    pytest.param(torch.half),
])
@pytest.mark.parametrize(['dst_device'], [
    pytest.param(torch.device('cpu')),
    pytest.param(torch.device('cuda', 0)),
])
@RunIf(min_gpus=1)
def test_submodules_device_and_dtype(dst_device, dst_dtype):
    """
    Test that the device and dtype property updates propagate through mixed nesting of regular
    nn.Modules and the special modules of type DeviceDtypeModuleMixin (e.g. Metric or LightningModule).
    """

    model = TopModule()
    assert model.device == torch.device('cpu')
    model = model.to(device=dst_device, dtype=dst_dtype)
    # nn.Module does not have these attributes
    assert not hasattr(model.module, '_device')
    assert not hasattr(model.module, '_dtype')
    # device and dtype change should propagate down into all children
    assert model.device == model.module.module.device == dst_device
    assert model.dtype == model.module.module.dtype == dst_dtype


@RunIf(min_gpus=2)
def test_submodules_multi_gpu_dp(tmpdir):
    model = TopModule()
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator='dp',
        gpus=2,
        callbacks=[DeviceAssertCallback()],
        max_steps=1,
    )
    trainer.fit(model)


@RunIf(min_gpus=2)
def test_submodules_multi_gpu_ddp_spawn(tmpdir):
    model = TopModule()
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator='ddp_spawn',
        gpus=2,
        callbacks=[DeviceAssertCallback()],
        max_steps=1,
    )
    trainer.fit(model)


@pytest.mark.parametrize(
    ['device'],
    [
        pytest.param(None),  # explicitly call without an index to see if the returning device contains an index
        pytest.param(0),
        pytest.param(torch.device('cuda', 0)),
    ]
)
@RunIf(min_gpus=1)
def test_gpu_cuda_device(device):
    model = TopModule()

    model.cuda(device)

    device = model.device
    assert device.type == 'cuda'
    assert device.index is not None
    assert device.index == torch.cuda.current_device()
