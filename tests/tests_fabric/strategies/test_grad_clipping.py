from contextlib import nullcontext
from unittest.mock import Mock

import pytest
import torch
from tests_fabric.helpers.models import BoringFabric
from tests_fabric.helpers.runif import RunIf

from lightning.fabric.accelerators.mps import MPSAccelerator
from lightning.fabric.strategies import DeepSpeedStrategy, FSDPStrategy
from lightning.fabric.wrappers import _FabricModule, _FabricOptimizer


class _MyFabricGradNorm(BoringFabric):
    def after_backward(self, model: _FabricModule, optimizer: _FabricOptimizer):
        self.clip_gradients(model, optimizer, max_norm=0.05, error_if_nonfinite=True)

        with model._forward_module.summon_full_params(model._forward_module) if isinstance(
            self.strategy, FSDPStrategy
        ) else nullcontext():
            parameters = model.parameters()
            grad_norm = torch.linalg.vector_norm(
                torch.stack([torch.linalg.vector_norm(p.grad.detach(), 2, dtype=torch.float32) for p in parameters]),
                2,
            )
        torch.testing.assert_close(grad_norm, torch.tensor(0.05, device=self.device))

    def run(self):
        while True:
            try:
                super().run()
                break
            except RuntimeError:  # nonfinite grads -> skip and continue
                pass


class _MyFabricGradVal(BoringFabric):
    def after_backward(self, model, optimizer):
        self.clip_gradients(model, optimizer, clip_val=1e-10)

        parameters = model.parameters()
        grad_max_list = [torch.max(p.grad.detach().abs()) for p in parameters]
        grad_max = torch.max(torch.stack(grad_max_list))
        torch.testing.assert_close(grad_max.abs(), torch.tensor(1e-10, device=self.device))


@pytest.mark.parametrize(
    "strategy,num_devices,precision",
    [
        pytest.param("auto", 1, "16-mixed"),
        pytest.param("ddp", 2, "32-true"),
        pytest.param("dp", 2, "16-mixed", marks=RunIf(min_cuda_gpus=2)),
        pytest.param(
            "fsdp",
            2,
            "32-true",
            marks=[RunIf(min_cuda_gpus=2), pytest.mark.xfail(reason="Testing with FSDP is not yet correct")],
        ),
        pytest.param(
            "fsdp",
            2,
            "16-true",
            marks=[RunIf(min_cuda_gpus=2), pytest.mark.xfail(reason="Testing with FSDP is not yet correct")],
        ),
    ],
)
@RunIf(standalone=True)
def test_grad_clipping_norm(strategy, num_devices, precision):
    accelerator = "cpu" if MPSAccelerator.is_available() else "auto"
    if accelerator == "cpu" and precision == "16-mixed":
        precision = "bf16-mixed"
    fabric = _MyFabricGradNorm(accelerator=accelerator, strategy=strategy, devices=num_devices, precision=precision)
    fabric.run()


@pytest.mark.parametrize(
    "strategy,num_devices,precision",
    [
        pytest.param("auto", 1, "16-mixed"),
        pytest.param("ddp", 2, "32-true"),
        pytest.param("dp", 2, "16-mixed", marks=RunIf(min_cuda_gpus=2)),
    ],
)
@RunIf(standalone=True)
def test_grad_clipping_val(strategy, num_devices, precision):
    accelerator = "cpu" if num_devices == 2 and torch.backends.mps.is_available() else "auto"
    if accelerator == "cpu" and precision == "16-mixed":
        precision = "bf16-mixed"
    fabric = _MyFabricGradVal(accelerator=accelerator, strategy=strategy, devices=num_devices, precision=precision)
    fabric.run()


@RunIf(deepspeed=True)
def test_errors_deepspeed():
    strategy = DeepSpeedStrategy()
    with pytest.raises(
        NotImplementedError,
        match=(
            "DeepSpeed handles gradient clipping automatically within the optimizer. "
            "Make sure to set the `gradient_clipping` value in your Config."
        ),
    ):
        strategy.clip_gradients_norm(Mock(), Mock(), Mock(), Mock(), Mock())

    with pytest.raises(
        NotImplementedError,
        match=(
            "DeepSpeed handles gradient clipping automatically within the optimizer. "
            "Make sure to set the `gradient_clipping` value in your Config."
        ),
    ):
        strategy.clip_gradients_value(Mock(), Mock(), Mock())


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
        strategy.clip_gradients_value(Mock(), Mock(), Mock())
