# Copyright The Lightning AI team.
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

from unittest import mock
from unittest.mock import ANY, MagicMock, Mock

import pytest
import torch
import torch.nn as nn
from torch.optim import Adam

from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.strategies.fsdp import _FSDPBackwardSyncControl
from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_1_12
from tests_fabric.helpers.runif import RunIf

if _TORCH_GREATER_EQUAL_1_12:
    from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, FullyShardedDataParallel, MixedPrecision
from tests_fabric.strategies.test_single_device import _MyFabricGradNorm


@mock.patch("lightning.fabric.strategies.fsdp._TORCH_GREATER_EQUAL_1_12", False)
def test_fsdp_support(*_):
    with pytest.raises(NotImplementedError, match="`FSDPStrategy` is supported from PyTorch v1.12.0"):
        FSDPStrategy()


@RunIf(min_torch="1.12")
def test_fsdp_custom_mixed_precision():
    """Test that passing a custom mixed precision config works."""
    config = MixedPrecision()
    strategy = FSDPStrategy(mixed_precision=config)
    assert strategy.mixed_precision_config == config


@RunIf(min_torch="1.12")
def test_fsdp_cpu_offload():
    """Test the different ways cpu offloading can be enabled."""
    # bool
    strategy = FSDPStrategy(cpu_offload=True)
    assert strategy.cpu_offload == CPUOffload(offload_params=True)

    # dataclass
    config = CPUOffload()
    strategy = FSDPStrategy(cpu_offload=config)
    assert strategy.cpu_offload == config


@RunIf(min_torch="1.12")
def test_fsdp_setup_optimizer_validation():
    """Test that `setup_optimizer()` validates the param groups and reference to FSDP parameters."""
    module = nn.Linear(2, 2)
    strategy = FSDPStrategy(parallel_devices=[torch.device("cpu")])

    bad_optimizer = Adam([{"params": [module.weight]}, {"params": [module.bias], "lr": 1e-3}])
    with pytest.raises(ValueError, match="does not support multiple param groups"):
        strategy.setup_optimizer(bad_optimizer)

    bad_optimizer = Adam(module.parameters())
    with pytest.raises(ValueError, match="The optimizer does not seem to reference any FSDP parameter"):
        strategy.setup_optimizer(bad_optimizer)


@RunIf(min_torch="1.12")
def test_fsdp_no_backward_sync():
    """Test that the backward sync control calls `.no_sync()`, and only on a module wrapped in
    FullyShardedDataParallel."""

    strategy = FSDPStrategy()
    assert isinstance(strategy._backward_sync_control, _FSDPBackwardSyncControl)

    with pytest.raises(
        TypeError, match="is only possible if the module passed to .* is wrapped in `FullyShardedDataParallel`"
    ):
        with strategy._backward_sync_control.no_backward_sync(Mock()):
            pass

    module = MagicMock(spec=FullyShardedDataParallel)
    with strategy._backward_sync_control.no_backward_sync(module):
        pass

    module.no_sync.assert_called_once()


@RunIf(min_torch="1.12")
@mock.patch("lightning.fabric.strategies.fsdp._TORCH_GREATER_EQUAL_1_13", False)
def test_fsdp_activation_checkpointing_support():
    """Test that we error out if activation checkpointing requires a newer PyTorch version."""
    with pytest.raises(ValueError, match="Activation checkpointing requires torch >= 1.13.0"):
        FSDPStrategy(activation_checkpointing=Mock())


@RunIf(min_torch="1.13")
def test_fsdp_activation_checkpointing():
    """Test that the FSDP strategy can apply activation checkpointing to the given layers."""

    class Block1(nn.Linear):
        pass

    class Block2(nn.Linear):
        pass

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer0 = nn.Sequential(Block1(4, 4), Block1(5, 5))
            self.layer1 = Block2(2, 2)
            self.layer2 = nn.Linear(3, 3)

    strategy = FSDPStrategy(activation_checkpointing=Block1)
    assert strategy._activation_checkpointing == [Block1]

    strategy = FSDPStrategy(activation_checkpointing=[Block1, Block2])
    assert strategy._activation_checkpointing == [Block1, Block2]

    strategy._parallel_devices = [torch.device("cuda", 0)]
    with mock.patch(
        "torch.distributed.fsdp.fully_sharded_data_parallel.FullyShardedDataParallel"
    ) as fsdp_mock, mock.patch(
        "torch.distributed.algorithms._checkpoint.checkpoint_wrapper.apply_activation_checkpointing"
    ) as ckpt_mock:
        strategy.setup_module(Model())
        ckpt_mock.assert_called_with(fsdp_mock(), checkpoint_wrapper_fn=ANY, check_fn=ANY)


@RunIf(min_torch="1.13")
def test_fsdp_grad_clipping_value_error():
    strategy = FSDPStrategy()
    with pytest.raises(
        NotImplementedError,
        match=(
            "FSDP currently does not support to clip gradients by value. "
            "Consider clipping by norm instead or choose another strategy!"
        ),
    ):
        strategy.clip_gradients_value(Mock(), Mock(), Mock())


class _MyFSDPFabricGradientNorm(_MyFabricGradNorm):
    def after_backward(self, model, optimizer):
        self.clip_gradients(model, optimizer, max_norm=0.05, error_if_nonfinite=True)

        with model._forward_module.summon_full_params(model._forward_module):
            parameters = model.parameters()
            grad_norm = torch.linalg.vector_norm(
                torch.stack([torch.linalg.vector_norm(p.grad.detach(), 2, dtype=torch.float32) for p in parameters]),
                2,
            )
            torch.testing.assert_close(grad_norm, torch.tensor(0.05, device=self.device))


@pytest.mark.parametrize(
    "precision",
    ["32-true", "16-mixed", pytest.param("bf16-mixed", marks=RunIf(bf16_cuda=True))],
)
@RunIf(min_cuda_gpus=2, standalone=True)
@pytest.mark.xfail(reason="Testing with FSDP is not yet correct")  # TODO: Investigate testing with fsdp
def test_fsdp_grad_clipping_norm(precision):
    fabric = _MyFSDPFabricGradientNorm(accelerator="cuda", devices=2, precision=precision, strategy="fsdp")
    fabric.run()
