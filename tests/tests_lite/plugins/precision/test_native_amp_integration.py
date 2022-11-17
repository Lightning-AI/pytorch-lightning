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
"""Integration tests for native automatic mixed precision (AMP) training."""
import pytest
import torch
import torch.nn as nn
from tests_lite.helpers.models import BoringLite
from tests_lite.helpers.runif import RunIf

from lightning_lite import LightningLite, seed_everything


class NativeMixedPrecisionModule(nn.Module):
    def __init__(self, expected_dtype):
        super().__init__()
        self.expected_dtype = expected_dtype
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        assert x.dtype == self.expected_dtype
        if x.device.type == "cpu":
            assert torch.is_autocast_cpu_enabled()
        else:
            assert torch.is_autocast_enabled()
        output = self.layer(x)
        assert output.dtype == self.expected_dtype
        return output


class NativeMixedPrecisionBoringLite(BoringLite):

    expected_dtype: torch.dtype

    def get_model(self):
        return NativeMixedPrecisionModule(self.expected_dtype)

    def step(self, model, batch):
        assert model.layer.weight.dtype == torch.float32

        assert batch.dtype == torch.float32
        output = model(batch)
        assert output.dtype == torch.float32
        loss = torch.nn.functional.mse_loss(output, torch.ones_like(output))
        return loss

    def after_backward(self, model):
        assert model.layer.weight.grad.dtype == torch.float32


@pytest.mark.parametrize(
    "accelerator, precision, expected_dtype",
    [
        ("cpu", 16, torch.bfloat16),
        ("cpu", "bf16", torch.bfloat16),
        pytest.param("cuda", 16, torch.float16, marks=RunIf(min_cuda_gpus=1)),
        pytest.param("cuda", "bf16", torch.bfloat16, marks=RunIf(min_cuda_gpus=1, bf16_cuda=True)),
    ],
)
def test_native_mixed_precision(accelerator, precision, expected_dtype):
    lite = NativeMixedPrecisionBoringLite(accelerator=accelerator, precision=precision)
    lite.expected_dtype = expected_dtype
    lite.run()


@RunIf(min_torch="1.13", min_cuda_gpus=1)
def test_native_mixed_precision_fused_optimizer_parity():
    def run(fused=False):
        seed_everything(1234)
        lite = LightningLite(accelerator="cuda", precision=16, devices=1)

        model = nn.Linear(10, 10).to(lite.device)  # TODO: replace with individual setup_model call
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0, fused=fused)

        model, optimizer = lite.setup(model, optimizer)
        assert isinstance(lite._precision.scaler, torch.cuda.amp.GradScaler)

        data = torch.randn(10, 10, device="cuda")
        target = torch.randn(10, 10, device="cuda")

        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            output = model(data)
            loss = (output - target).abs().sum()
            lite.backward(loss)
            optimizer.step()
            losses.append(loss.detach())
        return torch.stack(losses), model.parameters()

    losses, params = run(fused=False)
    losses_fused, params_fused = run(fused=True)

    # Both the regular and the fused version of Adam produce the same losses and model weights
    torch.testing.assert_close(losses, losses_fused)
    for p, q in zip(params, params_fused):
        torch.testing.assert_close(p, q)
