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
from unittest.mock import ANY, MagicMock, Mock

import lightning.fabric
import pytest
import torch
import torch.nn as nn
from lightning.fabric.plugins import HalfPrecision
from lightning.fabric.plugins.environments import LightningEnvironment
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.strategies.fsdp import (
    _FSDPBackwardSyncControl,
    _get_full_state_dict_context,
    _is_sharded_checkpoint,
)
from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_1, _TORCH_GREATER_EQUAL_2_2
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, FullyShardedDataParallel, MixedPrecision
from torch.optim import Adam


def test_custom_mixed_precision():
    """Test that passing a custom mixed precision config works."""
    config = MixedPrecision()
    strategy = FSDPStrategy(mixed_precision=config)
    assert strategy.mixed_precision_config == config


def test_cpu_offload():
    """Test the different ways cpu offloading can be enabled."""
    # bool
    strategy = FSDPStrategy(cpu_offload=True)
    assert strategy.cpu_offload == CPUOffload(offload_params=True)

    # dataclass
    config = CPUOffload()
    strategy = FSDPStrategy(cpu_offload=config)
    assert strategy.cpu_offload == config


def test_sharding_strategy():
    """Test the different ways the sharding strategy can be set."""
    from torch.distributed.fsdp import ShardingStrategy

    # default
    strategy = FSDPStrategy()
    assert strategy.sharding_strategy == ShardingStrategy.FULL_SHARD

    # enum
    strategy = FSDPStrategy(sharding_strategy=ShardingStrategy.SHARD_GRAD_OP)
    assert strategy.sharding_strategy == ShardingStrategy.SHARD_GRAD_OP

    # string
    strategy = FSDPStrategy(sharding_strategy="NO_SHARD")
    assert strategy.sharding_strategy == ShardingStrategy.NO_SHARD
    strategy = FSDPStrategy(sharding_strategy="no_shard")
    assert strategy.sharding_strategy == ShardingStrategy.NO_SHARD


@pytest.mark.parametrize("sharding_strategy", ["HYBRID_SHARD", "_HYBRID_SHARD_ZERO2"])
def test_hybrid_shard_configuration(sharding_strategy, monkeypatch):
    """Test that the hybrid sharding strategies can only be used with automatic wrapping or a manually specified pg."""
    with pytest.raises(RuntimeError, match="The hybrid sharding strategy requires you to pass at least one of"):
        FSDPStrategy(sharding_strategy=sharding_strategy)

    strategy = FSDPStrategy(auto_wrap_policy={nn.Linear}, sharding_strategy=sharding_strategy)
    assert strategy.sharding_strategy.name == sharding_strategy

    process_group = (Mock(), Mock())
    strategy = FSDPStrategy(sharding_strategy=sharding_strategy, process_group=process_group)
    assert strategy.sharding_strategy.name == sharding_strategy
    assert strategy._fsdp_kwargs["process_group"] is process_group

    monkeypatch.setattr("lightning.fabric.strategies.fsdp._TORCH_GREATER_EQUAL_2_2", False)
    with pytest.raises(ValueError, match="`device_mesh` argument is only supported in torch >= 2.2."):
        FSDPStrategy(device_mesh=Mock())

    monkeypatch.setattr("lightning.fabric.strategies.fsdp._TORCH_GREATER_EQUAL_2_2", True)
    device_mesh = Mock()
    strategy = FSDPStrategy(sharding_strategy=sharding_strategy, device_mesh=device_mesh)
    assert strategy.sharding_strategy.name == sharding_strategy
    assert strategy._fsdp_kwargs["device_mesh"] is device_mesh

    with pytest.raises(ValueError, match="process_group.* device_mesh=.* are mutually exclusive"):
        FSDPStrategy(sharding_strategy=sharding_strategy, process_group=process_group, device_mesh=device_mesh)


def test_checkpoint_io_unsupported():
    """Test that the FSDP strategy does not support the `CheckpointIO` plugin."""
    strategy = FSDPStrategy()
    with pytest.raises(NotImplementedError, match="does not use the `CheckpointIO` plugin"):
        _ = strategy.checkpoint_io

    with pytest.raises(NotImplementedError, match="does not support setting a `CheckpointIO` plugin"):
        strategy.checkpoint_io = Mock()


@mock.patch("lightning.fabric.strategies.fsdp.FSDPStrategy.setup_module")
def test_setup_use_orig_params(_):
    module = nn.Linear(2, 2)
    optimizer = Adam(module.parameters())

    strategy = FSDPStrategy(parallel_devices=[torch.device("cpu")], use_orig_params=False)
    assert not strategy._fsdp_kwargs["use_orig_params"]

    with pytest.raises(ValueError, match=r"`FSDPStrategy\(use_orig_params=False\)` but this is not supported"):
        strategy.setup_module_and_optimizers(module, optimizer)

    strategy = FSDPStrategy(parallel_devices=[torch.device("cpu")])
    assert strategy._fsdp_kwargs["use_orig_params"]
    strategy.setup_module_and_optimizers(module, optimizer)
    assert strategy._fsdp_kwargs["use_orig_params"]


def test_no_backward_sync():
    """Test that the backward sync control calls `.no_sync()`, and only on a module wrapped in
    FullyShardedDataParallel."""

    strategy = FSDPStrategy()
    assert isinstance(strategy._backward_sync_control, _FSDPBackwardSyncControl)

    with pytest.raises(
        TypeError, match="is only possible if the module passed to .* is wrapped in `FullyShardedDataParallel`"
    ), strategy._backward_sync_control.no_backward_sync(Mock(), True):
        pass

    module = MagicMock(spec=FullyShardedDataParallel)
    with strategy._backward_sync_control.no_backward_sync(module, False):
        pass
    module.no_sync.assert_not_called()
    with strategy._backward_sync_control.no_backward_sync(module, True):
        pass
    module.no_sync.assert_called_once()


def test_activation_checkpointing_support(monkeypatch):
    """Test that we error out if activation checkpointing requires a newer PyTorch version."""
    monkeypatch.setattr(lightning.fabric.strategies.fsdp, "_TORCH_GREATER_EQUAL_2_1", False)
    with pytest.raises(ValueError, match="activation_checkpointing_policy` requires torch >= 2.1.0"):
        FSDPStrategy(activation_checkpointing_policy=Mock())


def test_activation_checkpointing():
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

    if _TORCH_GREATER_EQUAL_2_1:
        from torch.distributed.fsdp.wrap import ModuleWrapPolicy

        strategy = FSDPStrategy(activation_checkpointing_policy={Block1})
        assert set(strategy._activation_checkpointing_kwargs) == {"auto_wrap_policy"}
        assert isinstance(strategy._activation_checkpointing_kwargs["auto_wrap_policy"], ModuleWrapPolicy)

        strategy = FSDPStrategy(activation_checkpointing_policy=ModuleWrapPolicy({Block1, Block2}))
        assert set(strategy._activation_checkpointing_kwargs) == {"auto_wrap_policy"}
        assert isinstance(strategy._activation_checkpointing_kwargs["auto_wrap_policy"], ModuleWrapPolicy)
    else:
        strategy = FSDPStrategy(activation_checkpointing=Block1)
        assert set(strategy._activation_checkpointing_kwargs) == {"check_fn"}

        strategy = FSDPStrategy(activation_checkpointing=[Block1, Block2])
        assert set(strategy._activation_checkpointing_kwargs) == {"check_fn"}

        strategy = FSDPStrategy(activation_checkpointing_policy={Block1})
        assert set(strategy._activation_checkpointing_kwargs) == {"check_fn"}

        strategy = FSDPStrategy(activation_checkpointing_policy={Block1, Block2})
        assert set(strategy._activation_checkpointing_kwargs) == {"check_fn"}

    strategy._parallel_devices = [torch.device("cuda", 0)]
    with mock.patch("torch.distributed.fsdp.FullyShardedDataParallel", new=MagicMock), mock.patch(
        "torch.distributed.algorithms._checkpoint.checkpoint_wrapper.apply_activation_checkpointing"
    ) as apply_mock:
        wrapped = strategy.setup_module(Model())
    apply_mock.assert_called_with(wrapped, checkpoint_wrapper_fn=ANY, **strategy._activation_checkpointing_kwargs)


def test_forbidden_precision_raises():
    with pytest.raises(TypeError, match="can only work with the `FSDPPrecision"):
        FSDPStrategy(precision=HalfPrecision())

    strategy = FSDPStrategy()
    with pytest.raises(TypeError, match="can only work with the `FSDPPrecision"):
        strategy.precision = HalfPrecision()


def test_grad_clipping_norm_error():
    strategy = FSDPStrategy()
    with pytest.raises(
        TypeError,
        match="only possible if the module.*is wrapped in `FullyShardedDataParallel`",
    ):
        strategy.clip_gradients_norm(Mock(), Mock(), Mock())


def test_save_checkpoint_storage_options(tmp_path):
    """Test that the FSDP strategy does not accept storage options for saving checkpoints."""
    strategy = FSDPStrategy()
    with pytest.raises(TypeError, match=escape("FSDPStrategy.save_checkpoint(..., storage_options=...)` is not")):
        strategy.save_checkpoint(path=tmp_path, state=Mock(), storage_options=Mock())


@mock.patch("lightning.fabric.strategies.fsdp.FSDPStrategy.broadcast", lambda _, x: x)
@mock.patch("lightning.fabric.strategies.fsdp._get_full_state_dict_context")
@mock.patch("lightning.fabric.strategies.fsdp._get_sharded_state_dict_context")
@mock.patch("lightning.fabric.strategies.fsdp.torch.save")
@mock.patch("lightning.fabric.strategies.fsdp.shutil")
def test_save_checkpoint_path_exists(shutil_mock, torch_save_mock, __, ___, tmp_path):
    strategy = FSDPStrategy(state_dict_type="full")

    # state_dict_type='full', path exists, path is not a sharded checkpoint: error
    path = tmp_path / "not-empty"
    path.mkdir()
    (path / "file").touch()
    assert not _is_sharded_checkpoint(path)
    with pytest.raises(IsADirectoryError, match="exists and is a directory"):
        strategy.save_checkpoint(path=path, state=Mock())

    # state_dict_type='full', path exists, path is a sharded checkpoint: no error (overwrite)
    path = tmp_path / "sharded-checkpoint"
    path.mkdir()
    (path / "meta.pt").touch()
    assert _is_sharded_checkpoint(path)
    model = Mock(spec=FullyShardedDataParallel)
    model.modules.return_value = [model]
    strategy.save_checkpoint(path=path, state={"model": model})
    shutil_mock.rmtree.assert_called_once_with(path)

    # state_dict_type='full', path exists, path is a file: no error (overwrite)
    path = tmp_path / "file.pt"
    path.touch()
    model = Mock(spec=FullyShardedDataParallel)
    model.modules.return_value = [model]
    torch_save_mock.reset_mock()
    strategy.save_checkpoint(path=path, state={"model": model})
    torch_save_mock.assert_called_once()

    strategy = FSDPStrategy(state_dict_type="sharded")
    save_mock = mock.patch(
        "torch.distributed.checkpoint.save"
        if _TORCH_GREATER_EQUAL_2_2
        else "torch.distributed.checkpoint.save_state_dict"
    )

    # state_dict_type='sharded', path exists, path is a folder: no error (overwrite)
    path = tmp_path / "not-empty-2"
    path.mkdir()
    (path / "file").touch()
    model = Mock(spec=FullyShardedDataParallel)
    model.modules.return_value = [model]
    with save_mock:
        strategy.save_checkpoint(path=path, state={"model": model})
    assert (path / "file").exists()

    # state_dict_type='sharded', path exists, path is a file: no error (overwrite)
    path = tmp_path / "file-2.pt"
    path.touch()
    model = Mock(spec=FullyShardedDataParallel)
    model.modules.return_value = [model]
    with save_mock:
        strategy.save_checkpoint(path=path, state={"model": model})
    assert path.is_dir()


def test_save_checkpoint_one_fsdp_module_required(tmp_path):
    """Test that the FSDP strategy can only save one FSDP model per checkpoint."""
    strategy = FSDPStrategy()

    # missing FSDP model
    with pytest.raises(ValueError, match="Could not find a FSDP model in the provided checkpoint state."):
        strategy.save_checkpoint(path=tmp_path, state={})
    with pytest.raises(ValueError, match="Could not find a FSDP model in the provided checkpoint state."):
        strategy.save_checkpoint(path=tmp_path, state={"model": torch.nn.Linear(3, 3)})

    # multiple FSDP models
    model1 = Mock(spec=FullyShardedDataParallel)
    model1.modules.return_value = [model1]
    model2 = Mock(spec=FullyShardedDataParallel)
    model2.modules.return_value = [model2]
    with pytest.raises(ValueError, match="Found multiple FSDP models in the given state."):
        strategy.save_checkpoint(path=tmp_path, state={"model1": model1, "model2": model2})


def test_load_checkpoint_no_state(tmp_path):
    """Test that the FSDP strategy can't load the full state without access to a model instance from the user."""
    strategy = FSDPStrategy()
    with pytest.raises(ValueError, match=escape("Got FSDPStrategy.load_checkpoint(..., state=None")):
        strategy.load_checkpoint(path=tmp_path, state=None)
    with pytest.raises(ValueError, match=escape("Got FSDPStrategy.load_checkpoint(..., state={})")):
        strategy.load_checkpoint(path=tmp_path, state={})


@mock.patch("lightning.fabric.strategies.fsdp.FSDPStrategy.broadcast", lambda _, x: x)
@mock.patch("lightning.fabric.strategies.model_parallel._lazy_load", Mock())
@mock.patch("lightning.fabric.strategies.model_parallel.torch.load", Mock())
def test_load_checkpoint_one_fsdp_module_required(tmp_path):
    """Test that the FSDP strategy can only load one FSDP model per checkpoint."""
    strategy = FSDPStrategy()

    # missing FSDP model
    with pytest.raises(ValueError, match="Could not find a FSDP model in the provided checkpoint state."):
        strategy.load_checkpoint(path=tmp_path, state={"other": "data"})
    with pytest.raises(ValueError, match="Could not find a FSDP model in the provided checkpoint state."):
        strategy.load_checkpoint(path=tmp_path, state={"model": torch.nn.Linear(3, 3)})

    # multiple FSDP models
    model1 = Mock(spec=FullyShardedDataParallel)
    model1.modules.return_value = [model1]
    model2 = Mock(spec=FullyShardedDataParallel)
    model2.modules.return_value = [model2]
    with pytest.raises(ValueError, match="Found multiple FSDP models in the given state."):
        strategy.load_checkpoint(path=tmp_path, state={"model1": model1, "model2": model2})

    # A raw nn.Module instead of a dictionary is ok
    model = Mock(spec=nn.Module)
    model.parameters.return_value = [torch.zeros(2, 1)]
    path = tmp_path / "full.ckpt"
    path.touch()
    strategy.load_checkpoint(path=path, state=model)


@mock.patch("lightning.fabric.strategies.fsdp.FSDPStrategy.broadcast", lambda _, x: x)
def test_save_checkpoint_unknown_state_dict_type(tmp_path):
    strategy = FSDPStrategy(state_dict_type="invalid")
    model = Mock(spec=FullyShardedDataParallel)
    model.modules.return_value = [model]
    with pytest.raises(ValueError, match="Unknown state_dict_type"):
        strategy.save_checkpoint(path=tmp_path, state={"model": model})


def test_load_unknown_checkpoint_type(tmp_path):
    """Test that the strategy validates the contents at the checkpoint path."""
    strategy = FSDPStrategy()
    model = Mock(spec=FullyShardedDataParallel)
    model.modules.return_value = [model]
    path = tmp_path / "empty_dir"  # neither a single file nor a directory with meta file
    path.mkdir()
    with pytest.raises(ValueError, match="does not point to a valid checkpoint"):
        strategy.load_checkpoint(path=path, state={"model": model})


def test_load_raw_checkpoint_validate_single_file(tmp_path):
    """Test that we validate the given checkpoint is a single file when loading a raw PyTorch state-dict checkpoint."""
    strategy = FSDPStrategy()
    model = Mock(spec=nn.Module)
    path = tmp_path / "folder"
    path.mkdir()
    with pytest.raises(ValueError, match="The given path must be a single file containing the full state dict"):
        strategy.load_checkpoint(path=path, state=model)


def test_load_raw_checkpoint_optimizer_unsupported(tmp_path):
    """Validate that the FSDP strategy does not yet support loading the raw PyTorch state-dict for an optimizer."""
    strategy = FSDPStrategy()
    optimizer = Mock(spec=torch.optim.Optimizer)
    with pytest.raises(
        NotImplementedError, match="Loading a single optimizer object from a checkpoint is not supported"
    ):
        strategy.load_checkpoint(path=tmp_path, state=optimizer)


@mock.patch("torch.distributed.init_process_group")
def test_set_timeout(init_process_group_mock):
    """Test that the timeout gets passed to the ``torch.distributed.init_process_group`` function."""
    test_timedelta = timedelta(seconds=30)
    strategy = FSDPStrategy(timeout=test_timedelta, parallel_devices=[torch.device("cpu")])
    strategy.cluster_environment = LightningEnvironment()
    strategy.accelerator = Mock()
    strategy.setup_environment()
    process_group_backend = strategy._get_process_group_backend()
    global_rank = strategy.cluster_environment.global_rank()
    world_size = strategy.cluster_environment.world_size()
    init_process_group_mock.assert_called_with(
        process_group_backend, rank=global_rank, world_size=world_size, timeout=test_timedelta
    )


@pytest.mark.parametrize("torch_ge_2_1", [True, False])
@mock.patch("torch.distributed.fsdp.fully_sharded_data_parallel.FullyShardedDataParallel.set_state_dict_type")
def test_get_full_state_dict_context_offload(set_type_mock, monkeypatch, torch_ge_2_1):
    """Test that the state dict context manager handles CPU offloading depending on the PyTorch version."""
    monkeypatch.setattr("lightning.fabric.strategies.fsdp._TORCH_GREATER_EQUAL_2_1", torch_ge_2_1)

    with _get_full_state_dict_context(module=Mock(spec=FullyShardedDataParallel), world_size=1):
        assert set_type_mock.call_args_list[0][0][2].offload_to_cpu is torch_ge_2_1  # model config
        assert set_type_mock.call_args_list[0][0][3].offload_to_cpu is torch_ge_2_1  # optim config

    set_type_mock.reset_mock()

    with _get_full_state_dict_context(module=Mock(spec=FullyShardedDataParallel), world_size=4):
        assert set_type_mock.call_args_list[0][0][2].offload_to_cpu  # model config
        assert set_type_mock.call_args_list[0][0][3].offload_to_cpu  # optim config
