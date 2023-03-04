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
import tempfile
from unittest import mock

import pytest
import torch

from lightning.fabric import Fabric
from lightning.fabric.plugins import FSDPPrecision
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_1_12, _TORCH_GREATER_EQUAL_2_0
from tests_fabric.helpers.models import BoringFabric
from tests_fabric.helpers.runif import RunIf

if _TORCH_GREATER_EQUAL_1_12:
    from torch.distributed.fsdp import FullyShardedDataParallel
    from torch.distributed.fsdp.wrap import wrap


def _get_model():
    return torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))


class _MyFabricManualWrapping(BoringFabric):
    def get_model(self):
        model = _get_model()
        for i, layer in enumerate(model):
            if i % 2 == 0:
                model[i] = wrap(layer)
        return model


class _MyFabric(BoringFabric):
    def get_model(self):
        return _get_model()


def _step(fabric, model, batch):
    forward_module = model._forward_module
    original_module = model.module
    assert isinstance(forward_module, FullyShardedDataParallel)
    assert isinstance(fabric._precision, FSDPPrecision)

    precision = torch.float16 if fabric._precision.precision == "16" else torch.bfloat16
    assert forward_module.mixed_precision.param_dtype == precision
    assert forward_module.mixed_precision.reduce_dtype == precision
    assert forward_module.mixed_precision.buffer_dtype == precision

    for layer_num in [0, 2]:
        assert isinstance(original_module[layer_num], FullyShardedDataParallel)
        assert original_module[layer_num].mixed_precision.param_dtype == precision
        assert original_module[layer_num].mixed_precision.reduce_dtype == precision
        assert original_module[layer_num].mixed_precision.buffer_dtype == precision

    output = model(batch)
    loss = torch.nn.functional.mse_loss(output, torch.ones_like(output))
    return loss


def _assert_save_equality(fabric, model, ckpt_path):
    current_state_dict = fabric._strategy.get_module_state_dict(model)

    checkpoint = fabric.load(ckpt_path)
    loaded_model = _get_model()
    loaded_model.load_state_dict(checkpoint)

    # model parameters are identical after loading
    for current_param, loaded_param in zip(current_state_dict.values(), loaded_model.state_dict().values()):
        assert torch.allclose(current_param.float().cpu(), loaded_param.cpu())


if _TORCH_GREATER_EQUAL_2_0:

    def _custom_auto_wrap_policy(
        module,
        recurse,
        nonwrapped_numel: int,
    ) -> bool:
        return nonwrapped_numel >= 2

else:

    def _custom_auto_wrap_policy(
        module,
        recurse,
        unwrapped_params: int,
    ) -> bool:
        return unwrapped_params >= 2


@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True, min_torch="1.13")
@pytest.mark.parametrize("precision", ("16-mixed", pytest.param("bf16-mixed", marks=RunIf(bf16_cuda=True))))
@pytest.mark.parametrize("manual_wrapping", [True, False])
def test_fsdp_train_save_load(manual_wrapping, precision):
    """Test FSDP training, saving and loading with different wrapping and precision settings."""
    strategy = FSDPStrategy(
        auto_wrap_policy=_custom_auto_wrap_policy,
        activation_checkpointing=[torch.nn.Linear],
    )
    fabric_cls = _MyFabricManualWrapping if manual_wrapping else _MyFabric
    fabric = fabric_cls(accelerator="cuda", strategy=strategy, devices=2, precision=precision)
    fabric.run()

    with tempfile.TemporaryFile() as ckpt_path:
        ckpt_path = fabric.broadcast(str(ckpt_path))
        fabric._strategy.save_checkpoint(ckpt_path, fabric.model.state_dict())

    _assert_save_equality(fabric, fabric.model, ckpt_path)


@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True, min_torch="1.12")
@pytest.mark.parametrize("move_to_device", [True, False])
@mock.patch("lightning.fabric.wrappers._FabricModule")
def test_setup_module_move_to_device(fabric_module_mock, move_to_device):
    """Test that `move_to_device` does nothing, FSDP decides which device parameters get moved to which device
    (sharding)."""
    strategy = FSDPStrategy(auto_wrap_policy=_custom_auto_wrap_policy)
    fabric = Fabric(accelerator="cuda", devices=2, strategy=strategy)
    fabric.launch()

    model = torch.nn.Linear(10, 10, bias=False)  # total params: 10 * 10 = 100
    fabric_model = fabric.setup_module(model, move_to_device=move_to_device)
    fabric_module_mock.assert_not_called()

    assert len(list(fabric_model.parameters())) == 1
    # the linear layer got sharded and each part is on the expected device
    assert next(fabric_model.parameters()).device == torch.device("cuda", fabric.local_rank)
    assert next(fabric_model.parameters()).numel() == 50

    # The _DeviceDtypeModuleMixin currently can't represent the device in a meaningful way for sharded models
    assert fabric_model.device == torch.device("cpu")
    assert fabric.device == torch.device("cuda", fabric.local_rank)
