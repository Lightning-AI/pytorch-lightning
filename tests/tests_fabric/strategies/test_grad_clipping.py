from unittest.mock import Mock

import pytest
import torch
from tests_fabric.helpers.models import BoringFabric
from tests_fabric.helpers.runif import RunIf

from lightning.fabric.strategies import DeepSpeedStrategy, FSDPStrategy


class _MyFabricGradNorm(BoringFabric):
    def after_backward(self, model, optimizer):
        self.strategy.clip_gradients_norm(
            optimizer,
            max_norm=0.05,
            error_if_nonfinite=False
        )
        parameters = model.parameters()
        grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2)
        torch.testing.assert_close(grad_norm, torch.tensor(0.05, device=self.device))


class _MyFabricGradVal(BoringFabric):
    def after_backward(self, model, optimizer):
        self.strategy.clip_gradients_value(optimizer, clip_val=1e-10)

        parameters = model.parameters()
        grad_max_list = [torch.max(p.grad.detach().abs()) for p in parameters]
        grad_max = torch.max(torch.stack(grad_max_list))
        torch.testing.assert_close(grad_max.abs(), torch.tensor(1e-10, device=self.device))


@pytest.mark.parametrize(
    "strategy,num_devices",
    [
        pytest.param(None, 1),
        pytest.param("ddp", 2),
        pytest.param("dp", 2),
        pytest.param("fsdp", 2, marks=RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True, min_torch="1.13")),
    ],
)
@pytest.mark.parametrize("precision", [16, 32])
@RunIf(standalone=True)
def test_grad_clipping_norm(strategy, num_devices, precision):
    accelerator = "cpu" if num_devices == 2 and torch.backends.mps.is_available() else "auto"
    fabric = _MyFabricGradNorm(accelerator=accelerator, strategy=strategy, devices=num_devices, precision=precision)
    fabric.run()


@pytest.mark.parametrize(
    "strategy,num_devices",
    [
        pytest.param(None, 1),
        pytest.param("ddp", 2),
        pytest.param("dp", 2),
        pytest.param("fsdp", 2, marks=RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True, min_torch="1.13")),
    ],
)
@pytest.mark.parametrize("precision", [16, 32])
@RunIf(standalone=True)
def test_grad_clipping_val(strategy, num_devices, precision):
    accelerator = "cpu" if num_devices == 2 and torch.backends.mps.is_available() else "auto"
    fabric = _MyFabricGradVal(accelerator=accelerator, strategy=strategy, devices=num_devices, precision=precision)
    fabric.run()


def test_errors_deepspeed():
    strategy = DeepSpeedStrategy()
    with pytest.raises(
        NotImplementedError,
        match=(
            "DeepSpeed handles gradient clipping automatically within the optimizer. "
            "Make sure to set the `gradient_clipping` value in your Config."
        ),
    ):
        strategy.clip_gradients_norm(Mock(), Mock(), Mock(), Mock())

    with pytest.raises(
        NotImplementedError,
        match=(
            "DeepSpeed handles gradient clipping automatically within the optimizer. "
            "Make sure to set the `gradient_clipping` value in your Config."
        ),
    ):
        strategy.clip_gradients_value(Mock(), Mock())


@RunIf(min_torch="1.13")
def test_fsdp_error():
    strategy = FSDPStrategy()
    with pytest.raises(
        NotImplementedError,
        match=(
            "FSDP currently does not support to clip gradients by value. "
            "Consider clipping by norm instead or choose another strategy!"
        ),
    ):
        strategy.clip_gradients_value(Mock(), Mock())
