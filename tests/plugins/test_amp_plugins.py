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
from pytorch_lightning.utilities import _TORCH_CPU_AMP_AVAILABLE
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers import BoringModel
from tests.helpers.runif import RunIf


class MyNativeAMP(NativeMixedPrecisionPlugin):
    pass


class MyApexPlugin(ApexMixedPrecisionPlugin):
    pass


@mock.patch.dict(
    os.environ,
    {
        "CUDA_VISIBLE_DEVICES": "0,1",
        "SLURM_NTASKS": "2",
        "SLURM_JOB_NAME": "SOME_NAME",
        "SLURM_NODEID": "0",
        "LOCAL_RANK": "0",
        "SLURM_PROCID": "0",
        "SLURM_LOCALID": "0",
    },
)
@mock.patch("torch.cuda.device_count", return_value=2)
@pytest.mark.parametrize("ddp_backend,gpus", [("ddp", 2), ("ddp2", 2), ("ddp_spawn", 2)])
@pytest.mark.parametrize(
    "amp,custom_plugin,plugin_cls",
    [
        pytest.param("native", False, NativeMixedPrecisionPlugin, marks=RunIf(amp_native=True)),
        pytest.param("native", True, MyNativeAMP, marks=RunIf(amp_native=True)),
        pytest.param("apex", False, ApexMixedPrecisionPlugin, marks=RunIf(amp_apex=True)),
        pytest.param("apex", True, MyApexPlugin, marks=RunIf(amp_apex=True)),
    ],
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
    if amp == "native":
        assert not trainer.precision_plugin.is_bfloat16


class GradientUnscaleBoringModel(BoringModel):
    def on_before_optimizer_step(self, *_):
        norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 2)
        if not (torch.isinf(norm) or torch.isnan(norm)):
            assert norm.item() < 15.0


@RunIf(min_gpus=2, amp_native=True)
@pytest.mark.parametrize("accum", [1, 2])
def test_amp_gradient_unscale(tmpdir, accum: int):
    model = GradientUnscaleBoringModel()

    trainer = Trainer(
        max_epochs=2,
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_test_batches=2,
        limit_val_batches=2,
        amp_backend="native",
        accelerator="ddp_spawn",
        gpus=2,
        precision=16,
        track_grad_norm=2,
        log_every_n_steps=1,
        accumulate_grad_batches=accum,
    )
    trainer.fit(model)


@RunIf(min_gpus=1, amp_native=True)
def test_amp_skip_optimizer(tmpdir):
    """
    Test that optimizers can be skipped when using amp
    """

    class CustomBoringModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.layer1 = torch.nn.Linear(32, 32)
            self.layer2 = torch.nn.Linear(32, 2)

        def forward(self, x: torch.Tensor):
            x = self.layer1(x)
            x = self.layer2(x)
            return x

        def training_step(self, batch, batch_idx, optimizer_idx):
            if optimizer_idx == 1:
                return None
            output = self(batch)
            return self.loss(batch, output)

        def configure_optimizers(self):
            return [
                torch.optim.SGD(self.layer1.parameters(), lr=0.1),
                torch.optim.SGD(self.layer2.parameters(), lr=0.1),
            ]

    trainer = Trainer(default_root_dir=tmpdir, gpus=1, fast_dev_run=1, amp_backend="native", precision=16)
    model = CustomBoringModel()
    trainer.fit(model)


@RunIf(min_gpus=2, amp_apex=True, special=True)
@pytest.mark.parametrize("amp_level", ["O2"])
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
        accelerator="ddp",
        plugins=ApexMixedPrecisionPlugin(amp_level=amp_level),
    )
    assert isinstance(trainer.precision_plugin, ApexMixedPrecisionPlugin)
    model = CustomBoringModel()
    trainer.fit(model)
    trainer.test(model)


@RunIf(min_gpus=2, amp_apex=True)
@pytest.mark.parametrize("amp_level", ["O2"])
def test_amp_apex_ddp_spawn_fit(amp_level, tmpdir):

    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        precision=16,
        amp_backend="apex",
        gpus=2,
        accelerator="ddp_spawn",
        plugins=ApexMixedPrecisionPlugin(amp_level=amp_level),
    )
    assert isinstance(trainer.precision_plugin, ApexMixedPrecisionPlugin)
    model = BoringModel()
    trainer.fit(model)


@RunIf(min_gpus=1, amp_native=True, max_torch="1.9")
def test_amp_precision_16_bfloat_throws_error(tmpdir):
    with pytest.raises(
        MisconfigurationException,
        match="To use bfloat16 with native amp you must install torch greater or equal to 1.10",
    ):
        Trainer(
            default_root_dir=tmpdir,
            precision="bf16",
            gpus=1,
        )


@RunIf(amp_native=True, max_torch="1.9")
def test_cpu_amp_precision_throws_error(tmpdir):
    with pytest.raises(
        MisconfigurationException,
        match="To use native AMP on CPU, install PyTorch 1.10 or later.",
    ):
        NativeMixedPrecisionPlugin(use_cpu=True)


@pytest.mark.skipif(not _TORCH_CPU_AMP_AVAILABLE, reason="Torch CPU AMP is not available.")
@RunIf(
    min_gpus=1,
    amp_native=True,
)
def test_cpu_amp_precision_context_manager(tmpdir):
    """
    Test to ensure that the context manager correctly is set to CPU + bfloat16, and a scaler isn't set.
    """

    plugin = NativeMixedPrecisionPlugin(precision="bf16", use_cpu=True)
    assert plugin.use_cpu
    assert not hasattr(plugin, "scaler")
    context_manager = plugin.autocast_context_manager()
    assert isinstance(context_manager, torch.cpu.amp.autocast)
    assert context_manager.dtype == torch.bfloat16


@pytest.mark.skipif(not _TORCH_CPU_AMP_AVAILABLE, reason="Torch CPU AMP is not available.")
@RunIf(
    min_gpus=1,
    amp_native=True,
)
def test_cpu_amp_precision_16_throws_error(tmpdir):
    """
    Throw error when using 16 as Native CPU AMP only supports bfloat16.
    """

    with pytest.raises(
        MisconfigurationException,
        match="CPU native amp only supports bfloat16. Please pass precision='bf16' to the Trainer.",
    ):
        Trainer(
            default_root_dir=tmpdir,
            precision=16,
        )
