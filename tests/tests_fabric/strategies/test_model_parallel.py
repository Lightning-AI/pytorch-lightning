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
from datetime import timedelta
from re import escape
from unittest import mock
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
from lightning.fabric.plugins.environments import LightningEnvironment
from lightning.fabric.strategies import ModelParallelStrategy
from lightning.fabric.strategies.model_parallel import _ParallelBackwardSyncControl
from torch.optim import Adam

from tests_fabric.helpers.runif import RunIf


@mock.patch("lightning.fabric.strategies.model_parallel._TORCH_GREATER_EQUAL_2_3", False)
def test_torch_greater_equal_2_3():
    with pytest.raises(ImportError, match="ModelParallelStrategy requires PyTorch 2.3 or higher"):
        ModelParallelStrategy(parallelize_fn=(lambda m, _: m))


@RunIf(min_torch="2.3")
def test_device_mesh_access():
    strategy = ModelParallelStrategy(parallelize_fn=(lambda m, _: m))
    with pytest.raises(RuntimeError, match="Accessing the device mesh .* not allowed"):
        _ = strategy.device_mesh


@RunIf(min_torch="2.3")
def test_checkpoint_io_unsupported():
    """Test that the ModelParallel strategy does not support the `CheckpointIO` plugin."""
    strategy = ModelParallelStrategy(parallelize_fn=(lambda m, _: m))
    with pytest.raises(NotImplementedError, match="does not use the `CheckpointIO` plugin"):
        _ = strategy.checkpoint_io

    with pytest.raises(NotImplementedError, match="does not support setting a `CheckpointIO` plugin"):
        strategy.checkpoint_io = Mock()


@RunIf(min_torch="2.3")
def test_save_filter_unsupported(tmp_path):
    strategy = ModelParallelStrategy(parallelize_fn=(lambda m, _: m))
    with pytest.raises(NotImplementedError, match="does not yet support the `filter` argument"):
        strategy.save_checkpoint(tmp_path / "checkpoint.pth", state={}, filter=Mock())


@RunIf(min_torch="2.3")
def test_load_raw_unsupported(tmp_path):
    strategy = ModelParallelStrategy(parallelize_fn=(lambda m, _: m))
    model = nn.Linear(2, 2)
    optimizer = Adam(model.parameters())
    with pytest.raises(NotImplementedError, match="object from a checkpoint directly is not yet supported"):
        strategy.load_checkpoint(tmp_path / "checkpoint.pth", state=model)
    with pytest.raises(NotImplementedError, match="object from a checkpoint directly is not yet supported"):
        strategy.load_checkpoint(tmp_path / "checkpoint.pth", state=optimizer)


@RunIf(min_torch="2.3")
def test_load_non_strict_unsupported(tmp_path):
    strategy = ModelParallelStrategy(parallelize_fn=(lambda m, _: m))
    with pytest.raises(NotImplementedError, match="Non-strict loading is not yet supported"):
        strategy.load_checkpoint(tmp_path / "checkpoint.pth", state={}, strict=False)


@RunIf(min_torch="2.3")
def test_parallelize_fn_call():
    model = nn.Linear(2, 2)
    optimizer = Adam(model.parameters())

    parallel_model_mock = Mock(spec=nn.Module, parameters=Mock(return_value=[]), buffers=Mock(return_value=[]))
    parallelize_fn = Mock(return_value=parallel_model_mock)
    strategy = ModelParallelStrategy(parallelize_fn=parallelize_fn)
    strategy._device_mesh = Mock()
    strategy.parallel_devices = [torch.device("cpu")]
    model_setup, [optimizer_setup] = strategy.setup_module_and_optimizers(model, [optimizer])
    assert model_setup is parallel_model_mock
    assert optimizer_setup is optimizer
    parallelize_fn.assert_called_with(model, strategy.device_mesh)

    # Raises an error if parallelize_fn does not return a module
    parallelize_fn = Mock(return_value=None)
    strategy = ModelParallelStrategy(parallelize_fn=parallelize_fn)
    strategy._device_mesh = Mock()
    strategy.parallel_devices = [torch.device("cpu")]
    with pytest.raises(TypeError, match="The `parallelize_fn` must return a `nn.Module` instance"):
        strategy.setup_module_and_optimizers(model, [optimizer])


@RunIf(min_torch="2.3")
def test_no_backward_sync():
    """Test that the backward sync control calls `.no_sync()`, and only on a module wrapped in
    FullyShardedDataParallel."""
    from torch.distributed._composable.fsdp import FSDP

    strategy = ModelParallelStrategy(parallelize_fn=(lambda m, _: m))
    assert isinstance(strategy._backward_sync_control, _ParallelBackwardSyncControl)

    fsdp_layer = Mock(spec=FSDP)
    other_layer = nn.Linear(2, 2)
    module = Mock()
    module.modules = Mock(return_value=[fsdp_layer, other_layer])

    with strategy._backward_sync_control.no_backward_sync(module, True):
        fsdp_layer.set_requires_gradient_sync.assert_called_with(False, recurse=False)
    fsdp_layer.set_requires_gradient_sync.assert_called_with(True, recurse=False)

    with strategy._backward_sync_control.no_backward_sync(module, False):
        fsdp_layer.set_requires_gradient_sync.assert_called_with(True, recurse=False)
    fsdp_layer.set_requires_gradient_sync.assert_called_with(False, recurse=False)


@RunIf(min_torch="2.3")
def test_save_checkpoint_storage_options(tmp_path):
    """Test that the FSDP strategy does not accept storage options for saving checkpoints."""
    strategy = ModelParallelStrategy(parallelize_fn=(lambda m, _: m))
    with pytest.raises(
        TypeError, match=escape("ModelParallelStrategy.save_checkpoint(..., storage_options=...)` is not")
    ):
        strategy.save_checkpoint(path=tmp_path, state=Mock(), storage_options=Mock())


@RunIf(min_torch="2.3")
@mock.patch("lightning.fabric.strategies.ModelParallelStrategy._setup_device_mesh")
@mock.patch("torch.distributed.init_process_group")
def test_set_timeout(init_process_group_mock, _):
    """Test that the timeout gets passed to the ``torch.distributed.init_process_group`` function."""
    test_timedelta = timedelta(seconds=30)
    strategy = ModelParallelStrategy(parallelize_fn=(lambda m, _: m), timeout=test_timedelta)
    strategy.parallel_devices = [torch.device("cpu")]
    strategy.cluster_environment = LightningEnvironment()
    strategy.accelerator = Mock()
    strategy.setup_environment()
    process_group_backend = strategy._get_process_group_backend()
    global_rank = strategy.cluster_environment.global_rank()
    world_size = strategy.cluster_environment.world_size()
    init_process_group_mock.assert_called_with(
        process_group_backend, rank=global_rank, world_size=world_size, timeout=test_timedelta
    )


@RunIf(min_torch="2.3")
def test_meta_device_materialization():
    """Test that the `setup_module()` method materializes meta-device tensors in the module."""

    class NoResetParameters(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(4, 4))

    class CustomModel(nn.Module):
        def __init__(self):
            super().__init__()
            # nn.Sequential as a parameterless module
            self.layer1 = nn.Sequential(NoResetParameters(), NoResetParameters())
            self.layer2 = nn.Linear(4, 4)
            self.register_buffer("buffer", torch.rand(2))

        def reset_parameters(self):
            self.buffer.fill_(1.0)

    strategy = ModelParallelStrategy(parallelize_fn=(lambda x, _: x))
    strategy._device_mesh = Mock()
    strategy._parallel_devices = [torch.device("cpu")]

    with torch.device("meta"):
        model = CustomModel()
    assert model.layer1[0].weight.is_meta
    assert model.layer2.weight.is_meta
    assert model.buffer.is_meta

    with pytest.warns(UserWarning, match=r"`reset_parameters\(\)` method for re-initialization: NoResetParameters"):
        model = strategy.setup_module(model)
    assert all(not p.is_meta for p in model.parameters())
    assert all(not b.is_meta for b in model.buffers())
