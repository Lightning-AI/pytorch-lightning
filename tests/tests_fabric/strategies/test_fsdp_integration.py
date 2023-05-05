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
import os
from copy import deepcopy
from unittest import mock

import pytest
import torch
from torch.nn import Parameter

from lightning.fabric import Fabric
from lightning.fabric.plugins import FSDPPrecision
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_1_12, _TORCH_GREATER_EQUAL_2_0
from lightning.fabric.wrappers import _FabricOptimizer
from tests_fabric.helpers.models import BoringFabric
from tests_fabric.helpers.runif import RunIf
from tests_fabric.test_fabric import BoringModel

if _TORCH_GREATER_EQUAL_1_12:
    from torch.distributed.fsdp import FlatParameter, FullyShardedDataParallel
    from torch.distributed.fsdp.wrap import always_wrap_policy, wrap


class _MyFabric(BoringFabric):
    def get_model(self):
        return torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))

    def step(self, model, batch):
        forward_module = model._forward_module
        original_module = model.module
        assert isinstance(forward_module, FullyShardedDataParallel)
        assert isinstance(self._precision, FSDPPrecision)

        precision = torch.float16 if self._precision.precision == "16-mixed" else torch.bfloat16
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


class _MyFabricManualWrapping(_MyFabric):
    def get_model(self):
        model = super().get_model()
        for i, layer in enumerate(model):
            if i % 2 == 0:
                model[i] = wrap(layer)
        return model


@RunIf(min_cuda_gpus=2, standalone=True, min_torch="2.0.0")
@pytest.mark.parametrize("precision", ["16-mixed", pytest.param("bf16-mixed", marks=RunIf(bf16_cuda=True))])
@pytest.mark.parametrize("manual_wrapping", [True, False])
def test_fsdp_train_save_load(tmp_path, manual_wrapping, precision):
    """Test FSDP training, saving and loading with different wrapping and precision settings."""
    fabric_cls = _MyFabricManualWrapping if manual_wrapping else _MyFabric
    fabric = fabric_cls(
        accelerator="cuda", strategy=FSDPStrategy(auto_wrap_policy=always_wrap_policy), devices=2, precision=precision
    )
    fabric.run()

    checkpoint_path = fabric.broadcast(str(tmp_path / "fsdp-checkpoint"))

    params_before = deepcopy(list(fabric.model.parameters()))
    state = {"model": fabric.model, "optimizer": fabric.optimizer, "steps": 1}
    fabric.save(checkpoint_path, state)
    assert set(os.listdir(checkpoint_path)) == {"meta.pt", ".metadata", "__0_0.distcp", "__1_0.distcp"}

    # re-init all objects and resume
    fabric = fabric_cls(
        accelerator="cuda", strategy=FSDPStrategy(auto_wrap_policy=always_wrap_policy), devices=2, precision=precision
    )
    fabric.run()

    # check correctness with loaded state
    state = {"model": fabric.model, "optimizer": fabric.optimizer, "steps": 0}
    metadata = fabric.load(checkpoint_path, state)
    params_after = deepcopy(list(fabric.model.parameters()))
    assert all(torch.equal(p0, p1) for p0, p1 in zip(params_before, params_after))

    # check user data in state reloaded
    assert state["steps"] == 1
    assert not metadata

    # attempt to load a key not in the metadata checkpoint
    state = {"model": fabric.model, "coconut": 11}
    with pytest.raises(KeyError, match="'coconut' not found in the checkpoint."):
        fabric.load(checkpoint_path, state)


@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True, min_torch="1.12")
@pytest.mark.parametrize("move_to_device", [True, False])
@mock.patch("lightning.fabric.wrappers._FabricModule")
def test_setup_module_move_to_device(fabric_module_mock, move_to_device):
    """Test that `move_to_device` does nothing, FSDP decides which device parameters get moved to which device
    (sharding)."""
    strategy = FSDPStrategy(auto_wrap_policy=always_wrap_policy)
    fabric = Fabric(accelerator="cuda", devices=2, strategy=strategy)
    fabric.launch()

    model = torch.nn.Linear(10, 10, bias=False)  # total params: 10 * 10 = 100
    fabric_model = fabric.setup_module(model, move_to_device=move_to_device)
    fabric_module_mock.assert_not_called()

    assert len(list(fabric_model.parameters())) == 1
    # the linear layer got sharded and each part is on the expected device
    assert next(fabric_model.parameters()).device == torch.device("cuda", fabric.local_rank)
    assert next(fabric_model.parameters()).numel() == 50
    if _TORCH_GREATER_EQUAL_2_0:
        # In PyTorch >= 2.0 we set `use_orig_params=True` and don't see flattened parameters
        assert isinstance(next(fabric_model.parameters()), Parameter)
    else:
        assert isinstance(next(fabric_model.parameters()), FlatParameter)

    # The _DeviceDtypeModuleMixin currently can't represent the device in a meaningful way for sharded models
    assert fabric_model.device == torch.device("cpu")
    assert fabric.device == torch.device("cuda", fabric.local_rank)


@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True, min_torch="2.0.0")
def test_setup_with_orig_params_and_multiple_param_groups():
    """Test that Fabric sets `use_orig_params` for the user when jointly setting up model and optimizer."""
    strategy = FSDPStrategy(auto_wrap_policy=always_wrap_policy)
    fabric = Fabric(accelerator="cuda", devices=2, strategy=strategy)
    fabric.launch()

    model = torch.nn.Sequential(
        torch.nn.Linear(10, 10, bias=False),
        torch.nn.Linear(5, 2, bias=False),
    )
    optimizer = torch.optim.Adam(
        [
            {"params": model[0].parameters(), "lr": 1e-2},
            {"params": model[1].parameters(), "lr": 1e-6},
        ]
    )

    # set up model and optimizer jointly
    wrapped_model, wrapped_optimizer = fabric.setup(model, optimizer)

    assert fabric.strategy._fsdp_kwargs["use_orig_params"]
    assert isinstance(wrapped_optimizer, _FabricOptimizer)
    assert len(wrapped_optimizer.param_groups) == 2
    for i in range(2):
        layer = wrapped_model._forward_module.module[i]
        assert isinstance(layer, FullyShardedDataParallel)
        assert torch.equal(wrapped_optimizer.param_groups[i]["params"][0], layer.weight)

        # A regular parameter as a view into the flattened parameters
        assert isinstance(layer.weight, torch.nn.Parameter)
        assert not isinstance(layer.weight, FlatParameter)


@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True, dynamo=True)
@mock.patch.dict(os.environ, {})
@pytest.mark.parametrize("compile_after_setup", [False, True])
def test_compile(compile_after_setup):
    """Test that the model can be compiled before and after the model is wrapped in FSDP."""
    model = BoringModel()
    strategy = FSDPStrategy(auto_wrap_policy=always_wrap_policy)
    fabric = Fabric(accelerator="cuda", devices=2, strategy=strategy)
    fabric.launch()

    if not compile_after_setup:
        model = torch.compile(model)

    model = fabric.setup(model)

    if compile_after_setup:
        model = torch.compile(model)

    for _ in range(3):
        model(torch.rand(2, 32, device=fabric.device)).sum().backward()


@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True)
@pytest.mark.parametrize(
    ("precision", "expected_dtype"),
    [
        ("32-true", torch.float32),
        ("16-true", torch.float16),
        pytest.param("bf16-true", torch.bfloat16, marks=RunIf(bf16_cuda=True)),
        ("64-true", torch.float64),
    ],
)
def test_module_init_context(precision, expected_dtype):
    """Test that the module under the init-context gets moved to the right device and dtype."""
    fabric = Fabric(
        accelerator="cuda",
        devices=2,
        strategy=FSDPStrategy(auto_wrap_policy=always_wrap_policy),
        precision=precision,
    )
    fabric.launch()

    with fabric.init_module():
        model = torch.nn.Linear(100, 100, bias=False)

    # The model is on the CPU until `.setup()``
    # TODO: Support initialization on meta device
    expected_device = torch.device("cpu")
    assert model.weight.device == expected_device
    assert model.weight.dtype == expected_dtype

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model, optimizer = fabric.setup(model, optimizer)

    # Parameters get sharded in `.setup()` and moved to the target device
    assert model.weight.device == torch.device("cuda", fabric.local_rank)
    assert model.weight.dtype == expected_dtype
