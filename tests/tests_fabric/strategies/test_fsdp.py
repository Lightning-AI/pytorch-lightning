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
import warnings
from datetime import timedelta
from re import escape
from unittest import mock
from unittest.mock import ANY, MagicMock, Mock

import pytest
import torch
import torch.nn as nn
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, FullyShardedDataParallel, MixedPrecision
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.optim import Adam

from lightning.fabric.plugins import HalfPrecision
from lightning.fabric.plugins.environments import LightningEnvironment
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.strategies.fsdp import (
    _FSDPBackwardSyncControl,
    _get_full_state_dict_context,
    _is_sharded_checkpoint,
    _warn_if_shared_params_across_fsdp_units,
)
from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_2, _TORCH_GREATER_EQUAL_2_3
from tests_fabric.helpers.runif import RunIf


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

    with (
        pytest.raises(
            TypeError, match="is only possible if the module passed to .* is wrapped in `FullyShardedDataParallel`"
        ),
        strategy._backward_sync_control.no_backward_sync(Mock(), True),
    ):
        pass

    module = MagicMock(spec=FullyShardedDataParallel)
    with strategy._backward_sync_control.no_backward_sync(module, False):
        pass
    module.no_sync.assert_not_called()
    with strategy._backward_sync_control.no_backward_sync(module, True):
        pass
    module.no_sync.assert_called_once()


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

    strategy = FSDPStrategy(activation_checkpointing_policy={Block1})
    assert set(strategy._activation_checkpointing_kwargs) == {"auto_wrap_policy"}
    assert isinstance(strategy._activation_checkpointing_kwargs["auto_wrap_policy"], ModuleWrapPolicy)

    strategy = FSDPStrategy(activation_checkpointing_policy=ModuleWrapPolicy({Block1, Block2}))
    assert set(strategy._activation_checkpointing_kwargs) == {"auto_wrap_policy"}
    assert isinstance(strategy._activation_checkpointing_kwargs["auto_wrap_policy"], ModuleWrapPolicy)

    strategy._parallel_devices = [torch.device("cuda", 0)]
    with (
        mock.patch("torch.distributed.fsdp.FullyShardedDataParallel", new=MagicMock),
        mock.patch(
            "torch.distributed.algorithms._checkpoint.checkpoint_wrapper.apply_activation_checkpointing"
        ) as apply_mock,
    ):
        wrapped = strategy.setup_module(Model())
    apply_mock.assert_called_with(wrapped, checkpoint_wrapper_fn=ANY, **strategy._activation_checkpointing_kwargs)


def test_setup_module_device_id_cpu():
    """``setup_module`` passes an explicit ``torch.device('cpu')`` (not ``device_id=None``) on CPU.

    ``root_device.index`` is ``None`` on CPU; ``device_id=None`` trips torch>=2.5's "FSDP needs a
    non-CPU accelerator device" guard. Only reachable when the GPU-accelerator guard is bypassed.

    """
    captured = {}

    class FakeFSDP(nn.Module):
        def __init__(self, module, **kwargs):
            super().__init__()
            captured.update(kwargs)
            self.module = module

    strategy = FSDPStrategy()
    strategy._parallel_devices = [torch.device("cpu")]
    with mock.patch("torch.distributed.fsdp.FullyShardedDataParallel", FakeFSDP):
        strategy.setup_module(nn.Linear(2, 2))
    assert captured["device_id"] == torch.device("cpu")


def test_module_sharded_context_device_id_cpu():
    """``module_sharded_context`` passes an explicit ``torch.device('cpu')`` (not ``device_id=None``) on CPU."""
    from contextlib import contextmanager

    captured = {}

    @contextmanager
    def fake_enable_wrap(*args, **kwargs):
        captured.update(kwargs)
        yield

    strategy = FSDPStrategy()
    strategy._parallel_devices = [torch.device("cpu")]
    with mock.patch("torch.distributed.fsdp.wrap.enable_wrap", fake_enable_wrap), strategy.module_sharded_context():
        pass
    assert captured["device_id"] == torch.device("cpu")


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
@mock.patch("lightning.fabric.strategies.fsdp._atomic_save")
@mock.patch("lightning.fabric.strategies.fsdp._remove_checkpoint")
def test_save_checkpoint_path_exists(remove_checkpoint_mock, atomic_save_mock, __, ___, tmp_path):
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
    remove_checkpoint_mock.assert_called_once_with(path)

    # state_dict_type='full', path exists, path is a file: no error (overwrite)
    path = tmp_path / "file.pt"
    path.touch()
    model = Mock(spec=FullyShardedDataParallel)
    model.modules.return_value = [model]
    atomic_save_mock.reset_mock()
    strategy.save_checkpoint(path=path, state={"model": model})
    atomic_save_mock.assert_called_once()

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
    kwargs = {}
    if _TORCH_GREATER_EQUAL_2_3:
        kwargs["device_id"] = strategy.root_device if strategy.root_device.type != "cpu" else None
    init_process_group_mock.assert_called_with(
        process_group_backend, rank=global_rank, world_size=world_size, timeout=test_timedelta, **kwargs
    )


@mock.patch("torch.distributed.fsdp.fully_sharded_data_parallel.FullyShardedDataParallel.set_state_dict_type")
def test_get_full_state_dict_context_offload(set_type_mock, monkeypatch):
    """Test that the state dict context manager only offloads to CPU when the shards live on an accelerator."""
    # Shards on accelerator: offload to CPU.
    accelerator_param = Mock()
    accelerator_param.device = torch.device("cuda", 0)
    module = Mock(spec=FullyShardedDataParallel)
    module.parameters = Mock(return_value=iter([accelerator_param]))
    with _get_full_state_dict_context(module=module, world_size=4):
        assert set_type_mock.call_args_list[0][0][2].offload_to_cpu  # model config
        assert set_type_mock.call_args_list[0][0][3].offload_to_cpu  # optim config

    set_type_mock.reset_mock()

    # Shards on CPU: do not offload (prevents PyTorch use-after-free).
    module = Mock(spec=FullyShardedDataParallel)
    module.parameters = Mock(return_value=iter([torch.nn.Parameter(torch.zeros(1))]))
    with _get_full_state_dict_context(module=module, world_size=4):
        assert not set_type_mock.call_args_list[0][0][2].offload_to_cpu  # model config
        assert not set_type_mock.call_args_list[0][0][3].offload_to_cpu  # optim config


def test_device_mesh_type_annotation():
    """Test that ``device_mesh`` type hint accepts a 2-element tuple via jsonargparse (#21580)."""
    jsonargparse = pytest.importorskip("jsonargparse")
    from inspect import signature

    annot = signature(FSDPStrategy).parameters["device_mesh"].annotation
    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--device_mesh", type=annot)
    args = parser.parse_args(["--device_mesh=[1, 4]"])
    assert args.device_mesh == (1, 4)


@RunIf(min_torch="2.2")
@mock.patch("torch.distributed.device_mesh.init_device_mesh")
@mock.patch("torch.distributed.init_process_group")
def test_device_mesh_initialization_cpu(init_process_group_mock, init_device_mesh_mock):
    """Test that device mesh is initialized with the correct device type on CPU."""
    strategy = FSDPStrategy(parallel_devices=[torch.device("cpu")], device_mesh=(2, 2))
    strategy.cluster_environment = LightningEnvironment()
    strategy.accelerator = Mock()
    strategy.setup_environment()

    init_device_mesh_mock.assert_called_with("cpu", (2, 2))


class _ModelWithTiedWeights(nn.Module):
    """A model that ties embedding and output head weights, similar to Llama/GPT-2."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, 32)
        self.layers = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32))
        self.head = nn.Linear(32, 100, bias=False)
        self.head.weight = self.embed.weight  # tie weights


def test_warn_shared_params_across_fsdp_units():
    """Test that a warning is emitted when tied weights would be split across FSDP units."""
    model = _ModelWithTiedWeights()
    policy = ModuleWrapPolicy({nn.Embedding})

    with pytest.warns(UserWarning, match="shared parameters"):
        _warn_if_shared_params_across_fsdp_units(model, policy)


def test_no_warn_shared_params_same_fsdp_unit():
    """Test that no warning is emitted when tied weights stay in the same FSDP unit."""
    model = _ModelWithTiedWeights()
    # Only wrap Sequential — both embed and head remain in root unit
    policy = ModuleWrapPolicy({nn.Sequential})

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_if_shared_params_across_fsdp_units(model, policy)


def test_no_warn_no_shared_params():
    """Test that no warning is emitted when the model has no shared parameters."""
    model = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2))
    policy = ModuleWrapPolicy({nn.Linear})

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_if_shared_params_across_fsdp_units(model, policy)


def test_no_warn_no_policy():
    """Test that no warning is emitted when no auto-wrap policy is set."""
    model = _ModelWithTiedWeights()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_if_shared_params_across_fsdp_units(model, None)


def test_is_full_checkpoint_remote_memory():
    import fsspec

    from lightning.fabric.strategies.fsdp import _is_full_checkpoint

    fs = fsspec.filesystem("memory")
    with fs.open("/c/full.ckpt", "wb") as f:
        f.write(b"x")
    assert _is_full_checkpoint("memory:///c/full.ckpt") is True
    assert _is_full_checkpoint("memory:///c/missing.ckpt") is False


def test_is_sharded_checkpoint_remote_memory():
    import fsspec

    from lightning.fabric.strategies.fsdp import _is_sharded_checkpoint

    fs = fsspec.filesystem("memory")
    with fs.open("/c2/sharded/meta.pt", "wb") as f:
        f.write(b"x")
    assert _is_sharded_checkpoint("memory:///c2/sharded") is True
    with fs.open("/c2/full.ckpt", "wb") as f:
        f.write(b"x")
    assert _is_sharded_checkpoint("memory:///c2/full.ckpt") is False


def test_distributed_checkpoint_reader_writer_selection(tmp_path):
    from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter

    from lightning.fabric.strategies.fsdp import (
        _get_distributed_checkpoint_reader,
        _get_distributed_checkpoint_writer,
    )

    assert isinstance(_get_distributed_checkpoint_writer(str(tmp_path)), FileSystemWriter)
    assert isinstance(_get_distributed_checkpoint_reader(str(tmp_path)), FileSystemReader)

    if _TORCH_GREATER_EQUAL_2_3:
        from torch.distributed.checkpoint._fsspec_filesystem import FsspecReader, FsspecWriter
    else:
        fsspec_fs = pytest.importorskip("torch.distributed.checkpoint._fsspec_filesystem")
        FsspecReader, FsspecWriter = fsspec_fs.FsspecReader, fsspec_fs.FsspecWriter

    assert isinstance(_get_distributed_checkpoint_writer("memory:///w/ckpt"), FsspecWriter)
    assert isinstance(_get_distributed_checkpoint_reader("memory:///w/ckpt"), FsspecReader)


def test_save_checkpoint_does_not_corrupt_remote_path(monkeypatch):
    """Regression: a gs:// URL must reach the DCP layer uncorrupted (not gs:/)."""
    strategy = FSDPStrategy()
    strategy._state_dict_type = "sharded"
    monkeypatch.setattr(strategy, "broadcast", lambda x: x)
    monkeypatch.setattr("lightning.fabric.strategies.fsdp._has_fsdp_modules", lambda m: True)

    captured = {}
    monkeypatch.setattr(
        "lightning.fabric.strategies.fsdp._distributed_checkpoint_save",
        lambda state, path: captured.update(path=path),
    )
    monkeypatch.setattr("lightning.fabric.strategies.fsdp._prepare_directory_checkpoint", lambda p: None)
    monkeypatch.setattr("lightning.fabric.strategies.fsdp._is_checkpoint_dir", lambda p: False)
    monkeypatch.setattr("lightning.fabric.strategies.fsdp._atomic_save", lambda obj, path: None)
    monkeypatch.setattr(
        "lightning.fabric.strategies.fsdp._get_sharded_state_dict_context",
        lambda module: mock.MagicMock(),
    )
    model = nn.Linear(2, 2)
    strategy.save_checkpoint("gs://bucket/run/ckpt", state={"model": model})
    assert captured["path"] == "gs://bucket/run/ckpt"


def test_load_raw_module_state_from_path_remote(monkeypatch):
    """Regression: remote full-checkpoints must be read via `_load`, not mmap/_lazy_load."""
    from lightning.fabric.strategies.model_parallel import _load_raw_module_state_from_path

    called = {}
    monkeypatch.setattr("lightning.fabric.strategies.model_parallel._is_full_checkpoint", lambda p: True)
    monkeypatch.setattr(
        "lightning.fabric.strategies.model_parallel._load_raw_module_state",
        lambda **kwargs: called.update(loaded=True),
    )
    monkeypatch.setattr(
        "lightning.fabric.strategies.model_parallel._load",
        lambda path, map_location=None: called.update(via_load=str(path)),
    )
    monkeypatch.setattr(
        "lightning.fabric.strategies.model_parallel._lazy_load",
        lambda path: called.update(via_lazy=str(path)),
    )
    _load_raw_module_state_from_path("memory:///x/full.ckpt", module=nn.Linear(2, 2), world_size=1)
    assert called.get("via_load") == "memory:///x/full.ckpt"
    assert "via_lazy" not in called


def test_load_full_checkpoint_remote_allows_non_tensor_objects(monkeypatch):
    """Regression: remote full-checkpoints are read with `weights_only=False` so non-tensor metadata
    (which `torch.load` rejects by default since torch 2.6) loads just like the local `_lazy_load` path."""
    strategy = FSDPStrategy()
    monkeypatch.setattr(strategy, "broadcast", lambda x: x)
    monkeypatch.setattr("lightning.fabric.strategies.fsdp._has_fsdp_modules", lambda m: True)
    monkeypatch.setattr("lightning.fabric.strategies.fsdp._is_full_checkpoint", lambda p: True)
    monkeypatch.setattr("lightning.fabric.strategies.model_parallel._load_raw_module_state", lambda *a, **k: None)

    captured = {}

    def fake_load(path, weights_only=None):
        captured["weights_only"] = weights_only
        return {"model": {"weight": torch.zeros(2)}}

    monkeypatch.setattr("lightning.fabric.strategies.fsdp._load", fake_load)

    strategy.load_checkpoint("memory:///x/full.ckpt", state={"model": nn.Linear(2, 2)})
    assert captured["weights_only"] is False


def test_load_full_checkpoint_remote_honors_explicit_weights_only(monkeypatch):
    """An explicit `weights_only=True` from the user must be honored for remote full-checkpoints, not silently
    overridden to `False`."""
    strategy = FSDPStrategy()
    monkeypatch.setattr(strategy, "broadcast", lambda x: x)
    monkeypatch.setattr("lightning.fabric.strategies.fsdp._has_fsdp_modules", lambda m: True)
    monkeypatch.setattr("lightning.fabric.strategies.fsdp._is_full_checkpoint", lambda p: True)
    monkeypatch.setattr("lightning.fabric.strategies.model_parallel._load_raw_module_state", lambda *a, **k: None)

    captured = {}

    def fake_load(path, weights_only=None):
        captured["weights_only"] = weights_only
        return {"model": {"weight": torch.zeros(2)}}

    monkeypatch.setattr("lightning.fabric.strategies.fsdp._load", fake_load)

    strategy.load_checkpoint("memory:///x/full.ckpt", state={"model": nn.Linear(2, 2)}, weights_only=True)
    assert captured["weights_only"] is True


def test_load_sharded_checkpoint_metadata_weights_only(monkeypatch):
    """The sharded-checkpoint metadata load must default to `weights_only=False` (like the full-checkpoint path) so
    non-tensor metadata loads on torch>=2.6, while still honoring an explicit user value."""
    strategy = FSDPStrategy()
    monkeypatch.setattr(strategy, "broadcast", lambda x: x)
    monkeypatch.setattr("lightning.fabric.strategies.fsdp._has_fsdp_modules", lambda m: True)
    monkeypatch.setattr("lightning.fabric.strategies.fsdp._is_sharded_checkpoint", lambda p: True)
    monkeypatch.setattr(
        "lightning.fabric.strategies.fsdp._get_sharded_state_dict_context", lambda module: mock.MagicMock()
    )
    monkeypatch.setattr("lightning.fabric.strategies.fsdp._distributed_checkpoint_load", lambda state, path: None)

    captured = {}

    def fake_load(path, weights_only=None):
        captured["weights_only"] = weights_only
        return {}

    monkeypatch.setattr("lightning.fabric.strategies.fsdp._load", fake_load)

    model = nn.Linear(2, 2)
    strategy.load_checkpoint("memory:///x/sharded", state={"model": model})
    assert captured["weights_only"] is False

    strategy.load_checkpoint("memory:///x/sharded", state={"model": model}, weights_only=True)
    assert captured["weights_only"] is True


def test_get_distributed_checkpoint_writer_missing_fsspec_module(monkeypatch):
    """A torch build without the private fsspec DCP module yields an actionable error, not a bare ImportError."""
    import sys

    from lightning.fabric.strategies.fsdp import _get_distributed_checkpoint_writer

    monkeypatch.setitem(sys.modules, "torch.distributed.checkpoint._fsspec_filesystem", None)
    with pytest.raises(ImportError, match=r"Remote .fsspec. distributed checkpoints require"):
        _get_distributed_checkpoint_writer("memory:///w/ckpt")


def test_get_distributed_checkpoint_reader_missing_fsspec_module(monkeypatch):
    """A torch build without the private fsspec DCP module yields an actionable error, not a bare ImportError."""
    import sys

    from lightning.fabric.strategies.fsdp import _get_distributed_checkpoint_reader

    monkeypatch.setitem(sys.modules, "torch.distributed.checkpoint._fsspec_filesystem", None)
    with pytest.raises(ImportError, match=r"Remote .fsspec. distributed checkpoints require"):
        _get_distributed_checkpoint_reader("memory:///w/ckpt")
