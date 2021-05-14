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

from pytorch_lightning import Trainer
from pytorch_lightning.plugins import ApexMixedPrecisionPlugin, NativeMixedPrecisionPlugin
from pytorch_lightning.plugins.precision import MixedPrecisionPlugin
from tests.helpers import BoringModel
from tests.helpers.runif import RunIf


class MyNativeAMP(NativeMixedPrecisionPlugin):
    pass


class MyApexPlugin(ApexMixedPrecisionPlugin):
    pass


@mock.patch.dict(
    os.environ, {
        "CUDA_VISIBLE_DEVICES": "0,1",
        "SLURM_NTASKS": "2",
        "SLURM_JOB_NAME": "SOME_NAME",
        "SLURM_NODEID": "0",
        "LOCAL_RANK": "0",
        "SLURM_PROCID": "0",
        "SLURM_LOCALID": "0",
    }
)
@mock.patch('torch.cuda.device_count', return_value=2)
@pytest.mark.parametrize('ddp_backend,gpus', [('ddp', 2), ('ddp2', 2), ('ddp_spawn', 2)])
@pytest.mark.parametrize(
    'amp,custom_plugin,plugin_cls', [
        pytest.param('native', False, NativeMixedPrecisionPlugin, marks=RunIf(amp_native=True)),
        pytest.param('native', True, MyNativeAMP, marks=RunIf(amp_native=True)),
        pytest.param('apex', False, ApexMixedPrecisionPlugin, marks=RunIf(amp_apex=True)),
        pytest.param('apex', True, MyApexPlugin, marks=RunIf(amp_apex=True)),
    ]
)
def test_amp_apex_ddp(
    mocked_device_count, ddp_backend: str, gpus: int, amp: str, custom_plugin: bool, plugin_cls: MixedPrecisionPlugin
):

    trainer = Trainer(
        fast_dev_run=True,
        precision=16,
        amp_backend=amp,
        gpus=gpus,
        accelerator=ddp_backend,
        plugins=[plugin_cls()] if custom_plugin else None,
    )
    assert isinstance(trainer.precision_plugin, plugin_cls)


class GradientUnscaleBoringModel(BoringModel):

    def on_after_backward(self):
        norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 2)
        if not (torch.isinf(norm) or torch.isnan(norm)):
            assert norm.item() < 15.


@RunIf(min_gpus=2, amp_native=True)
@pytest.mark.parametrize('accum', [1, 2])
def test_amp_gradient_unscale(tmpdir, accum: int):
    model = GradientUnscaleBoringModel()

    trainer = Trainer(
        max_epochs=2,
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_test_batches=2,
        limit_val_batches=2,
        amp_backend='native',
        accelerator='ddp_spawn',
        gpus=2,
        precision=16,
        track_grad_norm=2,
        log_every_n_steps=1,
        accumulate_grad_batches=accum,
    )
    trainer.fit(model)


@RunIf(min_gpus=2, amp_apex=True, special=True)
@pytest.mark.parametrize("amp_level", ['O2'])
def test_amp_apex_ddp_fit(amp_level, tmpdir):

    class CustomBoringModel(BoringModel):

        def training_step(self, batch, batch_idx):
            assert self.layer.weight.dtype == torch.float16
            assert self.trainer.precision_plugin._connected
            return super().training_step(batch, batch_idx)

    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        precision=16,
        amp_backend="apex",
        gpus=2,
        accelerator='ddp',
        plugins=ApexMixedPrecisionPlugin(amp_level=amp_level),
    )
    assert isinstance(trainer.precision_plugin, ApexMixedPrecisionPlugin)
    model = CustomBoringModel()
    trainer.fit(model)
    trainer.test(model)


@RunIf(min_gpus=2, amp_apex=True)
@pytest.mark.parametrize("amp_level", ['O2'])
def test_amp_apex_ddp_spawn_fit(amp_level, tmpdir):

    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        precision=16,
        amp_backend="apex",
        gpus=2,
        accelerator='ddp_spawn',
        plugins=ApexMixedPrecisionPlugin(amp_level=amp_level),
    )
    assert isinstance(trainer.precision_plugin, ApexMixedPrecisionPlugin)
    model = BoringModel()
    trainer.fit(model)
