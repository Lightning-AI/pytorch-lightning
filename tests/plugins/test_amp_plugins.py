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
@mock.patch("torch.cuda.is_available", return_value=True)
@mock.patch("torch.cuda.device_count", return_value=2)
@pytest.mark.parametrize("strategy,devices", [("ddp", 2), ("ddp2", 2), ("ddp_spawn", 2)])
@pytest.mark.parametrize(
    "amp,custom_plugin,plugin_cls",
    [
        ("native", False, NativeMixedPrecisionPlugin),
        ("native", True, MyNativeAMP),
        pytest.param("apex", False, ApexMixedPrecisionPlugin, marks=RunIf(amp_apex=True)),
        pytest.param("apex", True, MyApexPlugin, marks=RunIf(amp_apex=True)),
    ],
)
def test_amp_apex_ddp(mocked_device_count, mocked_is_available, strategy, devices, amp, custom_plugin, plugin_cls):
    plugin = None
    if custom_plugin:
        plugin = plugin_cls(16, "cpu") if amp == "native" else plugin_cls()
    trainer = Trainer(
        fast_dev_run=True,
        precision=16,
        amp_backend=amp,
        accelerator="gpu",
        devices=devices,
        strategy=strategy,
        plugins=plugin,
    )
    assert isinstance(trainer.precision_plugin, plugin_cls)


class TestClippingOptimizer(torch.optim.SGD):
    def step(self, *args, pl_module=None):
        pl_module.check_grads_clipped()
        return super().step(*args)


class TestPrecisionModel(BoringModel):
    # sister test: tests/trainer/optimization/test_manual_optimization.py::test_multiple_optimizers_step
    def on_after_backward(self) -> None:
        # check grads are scaled
        scale = self.trainer.precision_plugin.scaler.get_scale()
        assert scale != 1.0  # the return value if not enabled
        grads = [p.grad for p in self.parameters()]
        inv_scale = 1 / scale
        self.original_grads = [p * inv_scale for p in grads]

    def check_grads_unscaled(self, optimizer=None):
        if optimizer is not None:
            scaler = self.trainer.precision_plugin.scaler
            state = scaler._per_optimizer_states[id(optimizer)]
            assert state["stage"].name == "UNSCALED"

        grads = [p.grad for p in self.parameters()]
        assert len(grads) == len(self.original_grads)
        for actual, expected in zip(grads, self.original_grads):
            torch.testing.assert_allclose(actual, expected)

    def check_grads_clipped(self):
        parameters = list(self.parameters())
        assert len(parameters) == len(self.clipped_parameters)
        for actual, expected in zip(parameters, self.clipped_parameters):
            torch.testing.assert_allclose(actual.grad, expected.grad)

    def on_before_optimizer_step(self, optimizer, *_):
        self.check_grads_unscaled(optimizer)
        # manually clip
        self.clipped_parameters = []
        for p in self.parameters():
            copy = p.detach().clone()
            copy.grad = p.grad.clone()
            self.clipped_parameters.append(copy)
        clip_val = self.trainer.gradient_clip_val
        torch.nn.utils.clip_grad_value_(self.clipped_parameters, clip_val)

    def log_grad_norm(self, grad_norm_dict):
        self.check_grads_unscaled()
        assert len(grad_norm_dict)

    def configure_gradient_clipping(self, *args, **kwargs):
        # let lightning clip
        super().configure_gradient_clipping(*args, **kwargs)
        # check clipping worked as expected
        self.check_grads_clipped()

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, closure, **_):
        # pass self as a kwarg
        optimizer.step(closure, pl_module=self)

    def configure_optimizers(self):
        return TestClippingOptimizer(self.layer.parameters(), lr=0.1)


@RunIf(min_gpus=2)
@pytest.mark.parametrize("accum", [1, 2])
def test_amp_gradient_unscale(tmpdir, accum: int):
    model = TestPrecisionModel()

    trainer = Trainer(
        max_epochs=2,
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=0,
        amp_backend="native",
        strategy="ddp_spawn",
        accelerator="gpu",
        devices=2,
        precision=16,
        track_grad_norm=2,
        # use a tiny value to make sure it works
        gradient_clip_val=1e-3,
        gradient_clip_algorithm="value",
        log_every_n_steps=1,
        accumulate_grad_batches=accum,
        enable_progress_bar=False,
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

    trainer = Trainer(
        default_root_dir=tmpdir, accelerator="gpu", devices=1, fast_dev_run=1, amp_backend="native", precision=16
    )
    model = CustomBoringModel()
    trainer.fit(model)


@RunIf(min_gpus=2, amp_apex=True, standalone=True)
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
        accelerator="gpu",
        devices=2,
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
        accelerator="gpu",
        devices=2,
        strategy="ddp_spawn",
        plugins=ApexMixedPrecisionPlugin(amp_level=amp_level),
    )
    assert isinstance(trainer.precision_plugin, ApexMixedPrecisionPlugin)
    model = BoringModel()
    trainer.fit(model)


@RunIf(min_torch="1.10")
def test_cpu_amp_precision_context_manager(tmpdir):
    """Test to ensure that the context manager correctly is set to CPU + bfloat16."""
    plugin = NativeMixedPrecisionPlugin("bf16", "cpu")
    assert plugin.device == "cpu"
    assert plugin.scaler is None
    context_manager = plugin.autocast_context_manager()
    assert isinstance(context_manager, torch.autocast)
    # check with str due to a bug upstream: https://github.com/pytorch/pytorch/issues/65786
    assert str(context_manager.fast_dtype) == str(torch.bfloat16)


def test_precision_selection_raises(monkeypatch):
    with pytest.raises(
        MisconfigurationException, match=r"precision=16, amp_type='apex'\)` but apex AMP not supported on CPU"
    ):
        Trainer(amp_backend="apex", precision=16)

    import pytorch_lightning.plugins.precision.native_amp as amp

    monkeypatch.setattr(amp, "_TORCH_GREATER_EQUAL_1_10", False)
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
        with mock.patch("torch.cuda.is_available", return_value=True):
            Trainer(amp_backend="apex", precision=16, accelerator="gpu", devices=1, strategy="ddp_fully_sharded")

    import pytorch_lightning.plugins.precision.apex_amp as apex

    monkeypatch.setattr(apex, "_APEX_AVAILABLE", False)
    with mock.patch("torch.cuda.device_count", return_value=1), mock.patch(
        "torch.cuda.is_available", return_value=True
    ), pytest.raises(MisconfigurationException, match="asked for Apex AMP but you have not installed it"):
        Trainer(amp_backend="apex", precision=16, accelerator="gpu", devices=1)
