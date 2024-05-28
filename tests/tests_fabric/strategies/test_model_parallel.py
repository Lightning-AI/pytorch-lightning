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
from lightning.fabric.strategies.fsdp import _is_sharded_checkpoint
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
@pytest.mark.parametrize(
    ("num_nodes", "devices", "invalid_dp_size", "invalid_tp_size"),
    [
        (1, 4, 1, 1),
        (1, 4, 2, 3),
        (1, 4, 4, 2),
        (2, 4, 1, 4),
        (2, 4, 2, 1),
    ],
)
def test_validate_device_mesh_dimensions(num_nodes, devices, invalid_dp_size, invalid_tp_size):
    """Test passing sizes that don't multiply to the world size raises an error."""
    strategy = ModelParallelStrategy(
        parallelize_fn=(lambda m, _: m),
        data_parallel_size=invalid_dp_size,
        tensor_parallel_size=invalid_tp_size,
    )
    strategy._setup_distributed = Mock()
    strategy._accelerator = Mock()
    strategy.cluster_environment = Mock(
        world_size=Mock(return_value=(num_nodes * devices)), local_rank=Mock(return_value=1)
    )
    strategy.parallel_devices = [torch.device("cpu")] * devices
    strategy.num_nodes = num_nodes
    with pytest.raises(RuntimeError, match="multiplied should equal the world size"):
        strategy.setup_environment()


@RunIf(min_torch="2.3")
def test_checkpoint_io_unsupported():
    """Test that the ModelParallel strategy does not support the `CheckpointIO` plugin."""
    strategy = ModelParallelStrategy(parallelize_fn=(lambda m, _: m))
    with pytest.raises(NotImplementedError, match="does not use the `CheckpointIO` plugin"):
        _ = strategy.checkpoint_io

    with pytest.raises(NotImplementedError, match="does not support setting a `CheckpointIO` plugin"):
        strategy.checkpoint_io = Mock()


@RunIf(min_torch="2.3")
def test_fsdp_v1_modules_unsupported():
    """Test that the strategy won't allow setting up a module wrapped with the legacy FSDP API."""
    from torch.distributed.fsdp import FullyShardedDataParallel

    module = Mock(modules=Mock(return_value=[Mock(spec=FullyShardedDataParallel)]))
    strategy = ModelParallelStrategy(parallelize_fn=(lambda x, _: x))
    with pytest.raises(TypeError, match="only supports the new FSDP2 APIs in PyTorch >= 2.3"):
        strategy.setup_module(module)


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
    """Test that the backward sync control disables gradient sync on modules that benefit from it."""
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
    """Test that the strategy does not accept storage options for saving checkpoints."""
    strategy = ModelParallelStrategy(parallelize_fn=(lambda m, _: m))
    with pytest.raises(
        TypeError, match=escape("ModelParallelStrategy.save_checkpoint(..., storage_options=...)` is not")
    ):
        strategy.save_checkpoint(path=tmp_path, state=Mock(), storage_options=Mock())


@RunIf(min_torch="2.3")
@mock.patch("lightning.fabric.strategies.model_parallel.ModelParallelStrategy.broadcast", lambda _, x: x)
@mock.patch("lightning.fabric.strategies.model_parallel._has_dtensor_modules", return_value=True)
@mock.patch("torch.distributed.checkpoint.state_dict.get_model_state_dict", return_value={})
@mock.patch("torch.distributed.checkpoint.state_dict.get_optimizer_state_dict", return_value={})
@mock.patch("lightning.fabric.strategies.model_parallel.torch.save")
@mock.patch("lightning.fabric.strategies.model_parallel.shutil")
def test_save_checkpoint_path_exists(shutil_mock, torch_save_mock, _, __, ___, tmp_path):
    strategy = ModelParallelStrategy(parallelize_fn=(lambda m, _: m), save_distributed_checkpoint=False)

    # save_distributed_checkpoint=False, path exists, path is not a sharded checkpoint: error
    path = tmp_path / "not-empty"
    path.mkdir()
    (path / "file").touch()
    assert not _is_sharded_checkpoint(path)
    with pytest.raises(IsADirectoryError, match="exists and is a directory"):
        strategy.save_checkpoint(path=path, state=Mock())

    # save_distributed_checkpoint=False, path exists, path is a sharded checkpoint: no error (overwrite)
    path = tmp_path / "sharded-checkpoint"
    path.mkdir()
    (path / "meta.pt").touch()
    assert _is_sharded_checkpoint(path)
    model = Mock()
    model.modules.return_value = [model]
    strategy.save_checkpoint(path=path, state={"model": model})
    shutil_mock.rmtree.assert_called_once_with(path)

    # save_distributed_checkpoint=False, path exists, path is a file: no error (overwrite)
    path = tmp_path / "file.pt"
    path.touch()
    model = Mock(spec=nn.Module)
    torch_save_mock.reset_mock()
    strategy.save_checkpoint(path=path, state={"model": model})
    torch_save_mock.assert_called_once()

    strategy = ModelParallelStrategy(parallelize_fn=(lambda m, _: m), save_distributed_checkpoint=True)
    save_mock = mock.patch("torch.distributed.checkpoint.save")

    # save_distributed_checkpoint=True, path exists, path is a folder: no error (overwrite)
    path = tmp_path / "not-empty-2"
    path.mkdir()
    (path / "file").touch()
    model = Mock(spec=nn.Module)
    with save_mock:
        strategy.save_checkpoint(path=path, state={"model": model})
    assert (path / "file").exists()

    # save_distributed_checkpoint=True, path exists, path is a file: no error (overwrite)
    path = tmp_path / "file-2.pt"
    path.touch()
    model = Mock(spec=nn.Module)
    with save_mock:
        strategy.save_checkpoint(path=path, state={"model": model})
    assert path.is_dir()


@RunIf(min_torch="2.3")
def test_save_checkpoint_one_dist_module_required(tmp_path):
    """Test that the ModelParallelStrategy strategy can only save one distributed model per checkpoint."""
    strategy = ModelParallelStrategy(parallelize_fn=(lambda m, _: m))

    # missing DTensor model
    with pytest.raises(ValueError, match="Could not find a distributed model in the provided checkpoint state."):
        strategy.save_checkpoint(path=tmp_path, state={})
    with pytest.raises(ValueError, match="Could not find a distributed model in the provided checkpoint state."):
        strategy.save_checkpoint(path=tmp_path, state={"model": torch.nn.Linear(3, 3)})

    # multiple DTensor models
    with mock.patch("lightning.fabric.strategies.model_parallel._has_dtensor_modules", return_value=True):
        model1 = Mock(spec=nn.Module)
        model1.modules.return_value = [model1]
        model2 = Mock(spec=nn.Module)
        model2.modules.return_value = [model2]
        with pytest.raises(ValueError, match="Found multiple distributed models in the given state."):
            strategy.save_checkpoint(path=tmp_path, state={"model1": model1, "model2": model2})


@RunIf(min_torch="2.3")
@mock.patch("lightning.fabric.strategies.model_parallel.torch.load", Mock())
@mock.patch("lightning.fabric.strategies.model_parallel._TORCH_GREATER_EQUAL_2_4", False)
def test_load_full_checkpoint_support(tmp_path):
    """Test that loading non-distributed checkpoints into distributed models requires PyTorch >= 2.4."""
    strategy = ModelParallelStrategy(parallelize_fn=(lambda m, _: m))
    model = Mock(spec=nn.Module)
    model.parameters.return_value = [torch.zeros(2, 1)]
    path = tmp_path / "full.ckpt"
    path.touch()

    with pytest.raises(ImportError, match="Loading .* into a distributed model requires PyTorch >= 2.4"), mock.patch(
        "lightning.fabric.strategies.model_parallel._has_dtensor_modules", return_value=True
    ):
        strategy.load_checkpoint(path=path, state={"model": model})

    with pytest.raises(ImportError, match="Loading .* into a distributed model requires PyTorch >= 2.4"), mock.patch(
        "lightning.fabric.strategies.model_parallel._has_dtensor_modules", return_value=True
    ):
        strategy.load_checkpoint(path=path, state=model)


@RunIf(min_torch="2.3")
def test_load_checkpoint_no_state(tmp_path):
    """Test that the ModelParallelStrategy strategy can't load the full state without access to a model instance from
    the user."""
    strategy = ModelParallelStrategy(parallelize_fn=(lambda m, _: m))
    with pytest.raises(ValueError, match=escape("Got ModelParallelStrategy.load_checkpoint(..., state=None")):
        strategy.load_checkpoint(path=tmp_path, state=None)
    with pytest.raises(ValueError, match=escape("Got ModelParallelStrategy.load_checkpoint(..., state={})")):
        strategy.load_checkpoint(path=tmp_path, state={})


@RunIf(min_torch="2.3")
@mock.patch("lightning.fabric.strategies.model_parallel.ModelParallelStrategy.broadcast", lambda _, x: x)
@mock.patch("lightning.fabric.strategies.model_parallel.torch.load", Mock())
def test_load_checkpoint_one_dist_module_required(tmp_path):
    """Test that the ModelParallelStrategy strategy can only load one distributed model per checkpoint."""
    strategy = ModelParallelStrategy(parallelize_fn=(lambda m, _: m))

    # missing DTensor model
    with pytest.raises(ValueError, match="Could not find a distributed model in the provided checkpoint state."):
        strategy.load_checkpoint(path=tmp_path, state={"other": "data"})
    with pytest.raises(ValueError, match="Could not find a distributed model in the provided checkpoint state."):
        strategy.load_checkpoint(path=tmp_path, state={"model": torch.nn.Linear(3, 3)})

    # multiple DTensor models
    with mock.patch("lightning.fabric.strategies.model_parallel._has_dtensor_modules", return_value=True):
        model1 = Mock(spec=nn.Module)
        model1.modules.return_value = [model1]
        model2 = Mock(spec=nn.Module)
        model2.modules.return_value = [model2]
        with pytest.raises(ValueError, match="Found multiple distributed models in the given state."):
            strategy.load_checkpoint(path=tmp_path, state={"model1": model1, "model2": model2})

    # A raw nn.Module instead of a dictionary is ok
    model = Mock(spec=nn.Module)
    model.parameters.return_value = [torch.zeros(2, 1)]
    path = tmp_path / "full.ckpt"
    path.touch()
    strategy.load_checkpoint(path=path, state=model)


@RunIf(min_torch="2.3")
@mock.patch("lightning.fabric.strategies.model_parallel._has_dtensor_modules", return_value=True)
def test_load_unknown_checkpoint_type(_, tmp_path):
    """Test that the strategy validates the contents at the checkpoint path."""
    strategy = ModelParallelStrategy(parallelize_fn=(lambda m, _: m))
    model = Mock()
    path = tmp_path / "empty_dir"  # neither a single file nor a directory with meta file
    path.mkdir()
    with pytest.raises(ValueError, match="does not point to a valid checkpoint"):
        strategy.load_checkpoint(path=path, state={"model": model})


@RunIf(min_torch="2.3")
def test_load_raw_checkpoint_validate_single_file(tmp_path):
    """Test that we validate the given checkpoint is a single file when loading a raw PyTorch state-dict checkpoint."""
    strategy = ModelParallelStrategy(parallelize_fn=(lambda m, _: m))
    model = Mock(spec=nn.Module)
    path = tmp_path / "folder"
    path.mkdir()
    with pytest.raises(ValueError, match="The given path must be a single file containing the full state dict"):
        strategy.load_checkpoint(path=path, state=model)


@RunIf(min_torch="2.3")
def test_load_raw_checkpoint_optimizer_unsupported(tmp_path):
    """Validate that the ModelParallelStrategy strategy does not yet support loading the raw PyTorch state-dict for an
    optimizer."""
    strategy = ModelParallelStrategy(parallelize_fn=(lambda m, _: m))
    optimizer = Mock(spec=torch.optim.Optimizer)
    with pytest.raises(
        NotImplementedError, match="Loading a single optimizer object from a checkpoint is not supported"
    ):
        strategy.load_checkpoint(path=tmp_path, state=optimizer)


@RunIf(min_torch="2.3")
@mock.patch("lightning.fabric.strategies.model_parallel._setup_device_mesh")
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
