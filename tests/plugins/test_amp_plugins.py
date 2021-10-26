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
from pytorch_lightning.utilities import _TORCH_GREATER_EQUAL_DEV_1_10
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
        ("native", False, NativeMixedPrecisionPlugin),
        ("native", True, MyNativeAMP),
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
        strategy=ddp_backend,
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


@RunIf(min_gpus=2)
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
        strategy="ddp_spawn",
        gpus=2,
        precision=16,
        track_grad_norm=2,
        log_every_n_steps=1,
        accumulate_grad_batches=accum,
    )
    trainer.fit(model)


@RunIf(min_gpus=1)
def test_amp_skip_optimizer(tmpdir):
    """Test that optimizers can be skipped when using amp."""

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
        strategy="ddp",
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
        strategy="ddp_spawn",
        plugins=ApexMixedPrecisionPlugin(amp_level=amp_level),
    )
    assert isinstance(trainer.precision_plugin, ApexMixedPrecisionPlugin)
    model = BoringModel()
    trainer.fit(model)


@pytest.mark.skipif(not _TORCH_GREATER_EQUAL_DEV_1_10, reason="Torch CPU AMP is not available.")
def test_cpu_amp_precision_context_manager(tmpdir):
    """Test to ensure that the context manager correctly is set to CPU + bfloat16, and a scaler isn't set."""
    plugin = NativeMixedPrecisionPlugin(precision="bf16", use_cpu=True)
    assert plugin.use_cpu
    assert not hasattr(plugin, "scaler")
    context_manager = plugin.autocast_context_manager()
    assert isinstance(context_manager, torch.autocast)
    assert context_manager.fast_dtype == torch.bfloat16


def test_precision_selection_raises(monkeypatch):
    with pytest.raises(
        MisconfigurationException, match=r"precision=16, amp_type='apex'\)` but apex AMP not supported on CPU"
    ):
        Trainer(amp_backend="apex", precision=16)

    import pytorch_lightning.plugins.precision.native_amp as amp

    monkeypatch.setattr(amp, "_TORCH_GREATER_EQUAL_DEV_1_10", False)
    with pytest.warns(
        UserWarning, match=r"precision=16\)` but native AMP is not supported on CPU. Using `precision='bf16"
    ), pytest.raises(MisconfigurationException, match="must install torch greater or equal to 1.10"):
        Trainer(precision=16)

    with pytest.raises(MisconfigurationException, match="must install torch greater or equal to 1.10"):
        Trainer(precision="bf16")

    with pytest.raises(MisconfigurationException, match=r"amp_type='apex', precision='bf16'\)` but it's not supported"):
        Trainer(amp_backend="apex", precision="bf16")

    with mock.patch("torch.cuda.device_count", return_value=1), pytest.raises(
        MisconfigurationException, match="Sharded plugins are not supported with apex"
    ):
        Trainer(amp_backend="apex", precision=16, gpus=1, accelerator="ddp_fully_sharded")

    import pytorch_lightning.plugins.precision.apex_amp as apex

    monkeypatch.setattr(apex, "_APEX_AVAILABLE", False)
    with mock.patch("torch.cuda.device_count", return_value=1), pytest.raises(
        MisconfigurationException, match="asked for Apex AMP but you have not installed it"
    ):
        Trainer(amp_backend="apex", precision=16, gpus=1)
