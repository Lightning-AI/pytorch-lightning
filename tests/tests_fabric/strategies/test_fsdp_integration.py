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
from pathlib import Path
from unittest import mock
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
from lightning.fabric import Fabric
from lightning.fabric.plugins import FSDPPrecision
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_1
from lightning.fabric.utilities.load import _load_distributed_checkpoint
from lightning.fabric.wrappers import _FabricOptimizer
from torch._dynamo import OptimizedModule
from torch.distributed.fsdp import FlatParameter, FullyShardedDataParallel, OptimStateKeyType
from torch.distributed.fsdp.wrap import always_wrap_policy, wrap
from torch.nn import Parameter
from torch.utils.data import DataLoader

from tests_fabric.helpers.datasets import RandomDataset
from tests_fabric.helpers.runif import RunIf
from tests_fabric.test_fabric import BoringModel


class BasicTrainer:
    """Implements a basic training loop for the end-to-end tests."""

    def __init__(self, fabric):
        self.fabric = fabric
        self.model = self.optimizer = self.dataloader = None

    def get_model(self):
        return nn.Linear(32, 2)

    def step(self, model, batch):
        output = model(batch)
        return torch.nn.functional.mse_loss(output, torch.ones_like(output))

    def run(self) -> None:
        with self.fabric.init_module():
            model = self.get_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        model, optimizer = self.fabric.setup(model, optimizer)

        dataloader = DataLoader(RandomDataset(32, 64))
        dataloader = self.fabric.setup_dataloaders(dataloader)

        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader

        model.train()

        data_iter = iter(dataloader)
        batch = next(data_iter)
        loss = self.step(model, batch)
        self.fabric.backward(loss)
        optimizer.step()
        optimizer.zero_grad()


class _Trainer(BasicTrainer):
    def get_model(self):
        model = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
        self.num_wrapped = 4
        return model

    def step(self, model, batch):
        wrapped_layers = [m for m in model.modules() if isinstance(m, FullyShardedDataParallel)]
        assert len(wrapped_layers) == self.num_wrapped
        assert (self.num_wrapped == 4) == isinstance(model._forward_module, FullyShardedDataParallel)

        precision = self.fabric._precision
        assert isinstance(precision, FSDPPrecision)
        if precision.precision == "16-mixed":
            param_dtype = torch.float32
            reduce_dtype = buffer_dtype = torch.float16
        elif precision.precision == "bf16-mixed":
            param_dtype = torch.float32
            reduce_dtype = buffer_dtype = torch.bfloat16
        elif precision.precision == "16-true":
            param_dtype = reduce_dtype = buffer_dtype = torch.float16
        elif precision.precision == "bf16-true":
            param_dtype = reduce_dtype = buffer_dtype = torch.bfloat16
        else:
            raise ValueError(f"Unknown precision {precision.precision}")

        for layer in wrapped_layers:
            assert layer.mixed_precision.param_dtype == param_dtype
            assert layer.mixed_precision.reduce_dtype == reduce_dtype
            assert layer.mixed_precision.buffer_dtype == buffer_dtype

        output = model(batch)
        return torch.nn.functional.mse_loss(output, torch.ones_like(output))


class _TrainerManualWrapping(_Trainer):
    def get_model(self):
        model = super().get_model()
        for i, layer in enumerate(model):
            if i % 2 == 0:
                model[i] = wrap(layer)
        self.num_wrapped = 2
        return model


@RunIf(min_cuda_gpus=2, standalone=True)
@pytest.mark.parametrize("precision", ["16-mixed", pytest.param("bf16-mixed", marks=RunIf(bf16_cuda=True))])
@pytest.mark.parametrize("manual_wrapping", [True, False])
def test_train_save_load(tmp_path, manual_wrapping, precision):
    """Test FSDP training, saving and loading with different wrapping and precision settings."""
    trainer_cls = _TrainerManualWrapping if manual_wrapping else _Trainer
    fabric = Fabric(
        accelerator="cuda",
        strategy=FSDPStrategy(auto_wrap_policy=always_wrap_policy),
        devices=2,
        precision=precision,
    )
    fabric.launch()
    trainer = trainer_cls(fabric)
    trainer.run()

    checkpoint_path = fabric.broadcast(str(tmp_path / "fsdp-checkpoint"))

    params_before = deepcopy(list(trainer.model.parameters()))
    state = {"model": trainer.model, "optimizer": trainer.optimizer, "steps": 1}
    fabric.save(checkpoint_path, state)
    assert set(os.listdir(checkpoint_path)) == {"meta.pt", ".metadata", "__0_0.distcp", "__1_0.distcp"}

    # re-init all objects and resume
    fabric = Fabric(
        accelerator="cuda",
        strategy=FSDPStrategy(auto_wrap_policy=always_wrap_policy),
        devices=2,
        precision=precision,
    )
    fabric.launch()
    trainer = trainer_cls(fabric)
    trainer.run()

    # check correctness with loaded state
    state = {"model": trainer.model, "optimizer": trainer.optimizer, "steps": 0}
    metadata = fabric.load(checkpoint_path, state)
    for p0, p1 in zip(params_before, trainer.model.parameters()):
        torch.testing.assert_close(p0, p1, atol=0, rtol=0, equal_nan=True)

    # check user data in state reloaded
    assert state["steps"] == 1
    assert not metadata

    # attempt to load a key not in the metadata checkpoint
    state = {"model": trainer.model, "coconut": 11}
    with pytest.raises(KeyError, match="The requested state contains a key 'coconut' that does not exist"):
        fabric.load(checkpoint_path, state)

    # `strict=False` ignores the missing key
    state = {"model": trainer.model, "coconut": 11}
    fabric.load(checkpoint_path, state, strict=False)
    assert state["coconut"] == 11


@RunIf(min_cuda_gpus=2, standalone=True)
def test_save_full_state_dict(tmp_path):
    """Test that FSDP saves the full state into a single file with `state_dict_type="full"`."""
    fabric = Fabric(
        accelerator="cuda",
        strategy=FSDPStrategy(auto_wrap_policy=always_wrap_policy, state_dict_type="full"),
        devices=2,
    )
    fabric.launch()
    trainer = BasicTrainer(fabric)
    trainer.run()

    checkpoint_path = Path(fabric.broadcast(str(tmp_path / "fsdp-checkpoint.pt")))

    state = {"model": trainer.model, "optimizer": trainer.optimizer, "steps": 1}
    fabric.save(checkpoint_path, state)

    checkpoint = torch.load(checkpoint_path)
    assert checkpoint["steps"] == 1
    loaded_state_dict = checkpoint["model"]

    # assert the correct state model was saved
    with FullyShardedDataParallel.summon_full_params(trainer.model):
        state_dict = trainer.model.state_dict()
        assert set(loaded_state_dict.keys()) == set(state_dict.keys())
        for param_name in state_dict:
            assert torch.equal(loaded_state_dict[param_name], state_dict[param_name].cpu())
        params_before = [p.cpu() for p in trainer.model.parameters()]

    # assert the correct optimizer state was saved
    optimizer_state_before = FullyShardedDataParallel.full_optim_state_dict(
        trainer.model, trainer.optimizer, rank0_only=False
    )
    assert set(checkpoint["optimizer"].keys()) == set(optimizer_state_before.keys()) == {"state", "param_groups"}

    # 1. verify the FSDP state can be loaded back into a FSDP model/strategy directly
    fabric = Fabric(
        accelerator="cuda",
        strategy=FSDPStrategy(auto_wrap_policy=always_wrap_policy),
        devices=2,
    )
    fabric.launch()
    trainer = BasicTrainer(fabric)
    trainer.run()
    metadata = fabric.load(checkpoint_path, {"model": trainer.model, "optimizer": trainer.optimizer})
    assert metadata == {"steps": 1}

    with FullyShardedDataParallel.summon_full_params(trainer.model):
        params_after = list(trainer.model.parameters())
        assert all(torch.equal(p0.cpu(), p1.cpu()) for p0, p1 in zip(params_before, params_after))

    # assert the correct optimizer state was loaded
    optimizer_state_after = FullyShardedDataParallel.full_optim_state_dict(
        trainer.model, trainer.optimizer, rank0_only=False
    )
    assert set(optimizer_state_after.keys()) == set(optimizer_state_before.keys()) == {"state", "param_groups"}
    torch.testing.assert_close(optimizer_state_after["state"], optimizer_state_before["state"], atol=0, rtol=0)
    assert optimizer_state_after["param_groups"] == optimizer_state_before["param_groups"]

    # run a step to verify the optimizer state is correct
    trainer.run()

    # 2. verify the FSDP state can be loaded back into a single-device model/strategy
    fabric = Fabric(accelerator="cpu", devices=1)
    trainer = BasicTrainer(fabric)
    trainer.run()
    metadata = fabric.load(checkpoint_path, {"model": trainer.model, "optimizer": trainer.optimizer})
    assert metadata == {"steps": 1}
    params_after = list(trainer.model.parameters())
    assert all(torch.equal(p0, p1) for p0, p1 in zip(params_before, params_after))

    # get optimizer state after loading
    normal_checkpoint_path = Path(fabric.broadcast(str(tmp_path / "normal-checkpoint.pt")))
    fabric.save(normal_checkpoint_path, {"model": trainer.model, "optimizer": trainer.optimizer, "steps": 2})
    optimizer_state_after = torch.load(normal_checkpoint_path)["optimizer"]
    optimizer_state_after = FullyShardedDataParallel.rekey_optim_state_dict(
        optimizer_state_after, optim_state_key_type=OptimStateKeyType.PARAM_NAME, model=trainer.model
    )

    # assert the correct optimizer state was loaded
    assert set(optimizer_state_after.keys()) == set(optimizer_state_before.keys()) == {"state", "param_groups"}
    torch.testing.assert_close(optimizer_state_after["state"], optimizer_state_before["state"], atol=0, rtol=0)

    # run a step to verify the optimizer state is correct
    trainer.run()

    # 3. verify that a single-device model/strategy states can be loaded into a FSDP model/strategy
    fabric = Fabric(
        accelerator="cuda",
        strategy=FSDPStrategy(auto_wrap_policy=always_wrap_policy),
        devices=2,
    )
    fabric.launch()
    trainer = BasicTrainer(fabric)
    trainer.run()
    metadata = fabric.load(normal_checkpoint_path, {"model": trainer.model, "optimizer": trainer.optimizer})
    assert metadata == {"steps": 2}

    with FullyShardedDataParallel.summon_full_params(trainer.model):
        params_after = list(trainer.model.parameters())
        assert all(torch.equal(p0.cpu(), p1.cpu()) for p0, p1 in zip(params_before, params_after))

    # assert the correct optimizer state was loaded
    optimizer_state_after = FullyShardedDataParallel.full_optim_state_dict(
        trainer.model, trainer.optimizer, rank0_only=False
    )
    assert set(optimizer_state_after.keys()) == set(optimizer_state_before.keys()) == {"state", "param_groups"}
    torch.testing.assert_close(optimizer_state_after["state"], optimizer_state_before["state"], atol=0, rtol=0)
    assert optimizer_state_after["param_groups"] == optimizer_state_before["param_groups"]

    # run a step to verify the optimizer state is correct
    trainer.run()


@RunIf(min_cuda_gpus=2, standalone=True)
def test_load_full_state_dict_into_sharded_model(tmp_path):
    """Test that the strategy can load a full-state checkpoint into a FSDP sharded model."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    fabric = Fabric(accelerator="cuda", devices=1)
    fabric.seed_everything(0)
    trainer = BasicTrainer(fabric)
    trainer.run()

    # Save a full-state-dict checkpoint
    checkpoint_path = Path(fabric.broadcast(str(tmp_path / "full-checkpoint.pt")))
    state = {"model": trainer.model, "optimizer": trainer.optimizer, "steps": 1}
    fabric.save(checkpoint_path, state)

    # Gather all weights and store a copy manually
    with FSDP.summon_full_params(trainer.model, writeback=False, rank0_only=False):
        params_before = torch.cat([p.cpu().view(-1) for p in trainer.model.parameters()])

    # Create a FSDP sharded model
    fabric = Fabric(
        accelerator="cuda",
        strategy=FSDPStrategy(auto_wrap_policy=always_wrap_policy),
        devices=2,
    )
    fabric.launch()
    trainer = BasicTrainer(fabric)
    trainer.run()

    state = {"model": trainer.model, "optimizer": trainer.optimizer, "steps": 44}
    fabric.load(checkpoint_path, state)
    assert state["steps"] == 1

    # Gather all weights and compare
    with FSDP.summon_full_params(trainer.model, writeback=False, rank0_only=False):
        params_after = torch.cat([p.cpu().view(-1) for p in trainer.model.parameters()])
    assert torch.equal(params_before, params_after)

    # Create a raw state-dict checkpoint to test `Fabric.load_raw` too
    raw_checkpoint_path = checkpoint_path.with_name("model-state-dict")
    if fabric.global_rank == 0:
        checkpoint = torch.load(checkpoint_path)
        torch.save(checkpoint["model"], raw_checkpoint_path)
    fabric.barrier()

    trainer.run()
    fabric.load_raw(raw_checkpoint_path, trainer.model)

    # Gather all weights and compare
    with FSDP.summon_full_params(trainer.model, writeback=False, rank0_only=False):
        params_after = torch.cat([p.cpu().view(-1) for p in trainer.model.parameters()])
    assert torch.equal(params_before, params_after)


@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True)
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
    assert isinstance(next(fabric_model.parameters()), Parameter)

    # The _DeviceDtypeModuleMixin currently can't represent the device in a meaningful way for models with pieces on
    # different devices
    assert fabric_model.device == torch.device("cuda", fabric.local_rank)
    assert fabric.device == torch.device("cuda", fabric.local_rank)


@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True)
def test_setup_with_orig_params_and_multiple_param_groups():
    """Test that Fabric sets `use_orig_params` for the user when jointly setting up model and optimizer."""
    strategy = FSDPStrategy(auto_wrap_policy=always_wrap_policy)
    fabric = Fabric(accelerator="cuda", devices=2, strategy=strategy)
    fabric.launch()

    model = torch.nn.Sequential(
        torch.nn.Linear(10, 10, bias=False),
        torch.nn.Linear(5, 2, bias=False),
    )
    optimizer = torch.optim.Adam([
        {"params": model[0].parameters(), "lr": 1e-2},
        {"params": model[1].parameters(), "lr": 1e-6},
    ])

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


@RunIf(min_cuda_gpus=2, standalone=True, min_torch="2.1.0", dynamo=True, skip_windows=True)
@mock.patch("lightning.fabric.wrappers.torch.compile", Mock(wraps=torch.compile))
@mock.patch.dict(os.environ, {})
def test_reapply_compile():
    """Test that Fabric can rewrap a compiled module such that compilation happens over the FSDP-wrapper."""
    strategy = FSDPStrategy(auto_wrap_policy=always_wrap_policy)
    fabric = Fabric(accelerator="cuda", devices=2, strategy=strategy)
    fabric.launch()

    model = BoringModel()
    compile_kwargs = {"mode": "reduce-overhead"}
    compiled_model = torch.compile(model, **compile_kwargs)
    torch.compile.reset_mock()

    fabric_model = fabric.setup(compiled_model, _reapply_compile=True)

    assert isinstance(fabric_model._forward_module, OptimizedModule)
    assert isinstance(fabric_model._forward_module._orig_mod, FullyShardedDataParallel)

    # Assert we called compile again with the same arguments, but on the FSDP-wrapped module
    torch.compile.assert_called_with(fabric_model._forward_module._orig_mod, **compile_kwargs)

    assert fabric_model._original_module == model
    assert fabric_model._forward_module._orig_mod.module == model
    assert fabric_model.device == fabric.device

    # Smoke-testing forward to ensure we don't get compilation errors
    for _ in range(3):
        loss = fabric_model(torch.randn(2, 32, device=fabric.device)).sum()
        fabric.backward(loss)


@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True)
@pytest.mark.parametrize(
    ("precision", "expected_dtype"),
    [
        ("32-true", torch.float32),
        ("16-true", torch.float16),
        pytest.param("bf16-true", torch.bfloat16, marks=RunIf(bf16_cuda=True)),
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

    def _run_setup_assertions(empty_init, expected_device):
        with fabric.init_module(empty_init=empty_init):
            model = torch.nn.Linear(100, 100, bias=False)

        # The model is on the CPU/meta-device until after `.setup()``
        assert model.weight.device == expected_device
        assert model.weight.dtype == expected_dtype
        model = fabric.setup(model)
        # Parameters get sharded in `.setup()` and moved to the target device
        assert model.weight.device == torch.device("cuda", fabric.local_rank)
        assert model.weight.dtype == expected_dtype

    # Case 1: No empty init
    _run_setup_assertions(empty_init=False, expected_device=torch.device("cpu"))

    if _TORCH_GREATER_EQUAL_2_1:
        # Case 2: Empty-init with PyTorch >= 2.1 supports meta device
        _run_setup_assertions(empty_init=True, expected_device=torch.device("meta"))
    else:
        # Case 2: Empty-init with PyTorch < 2.1 only supports `torch.empty()`-init
        _run_setup_assertions(empty_init=True, expected_device=torch.device("cpu"))


@RunIf(min_cuda_gpus=2, standalone=True)
def test_save_filter(tmp_path):
    fabric = Fabric(accelerator="cuda", strategy=FSDPStrategy(state_dict_type="full"), devices=2)
    fabric.launch()
    model = nn.Linear(32, 2)
    model = fabric.setup_module(model)

    tmp_path = Path(fabric.broadcast(str(tmp_path)))
    state = {"model": model}
    filter = {"model": lambda k, v: "bias" in k}

    checkpoint_path = tmp_path / "full.pth"
    fabric.save(checkpoint_path, state, filter=filter)
    checkpoint = torch.load(checkpoint_path)["model"]
    assert set(checkpoint) == {"bias"}
    assert isinstance(checkpoint["bias"], torch.Tensor)

    fabric.strategy._state_dict_type = "sharded"
    checkpoint_path = tmp_path / "sharded"
    with pytest.raises(NotImplementedError, match="doesn't support loading sharded filtered"):
        fabric.save(checkpoint_path, state, filter=filter)


@RunIf(min_cuda_gpus=1)
def test_manual_activation_checkpointing():
    model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Linear(1, 1))
    strategy = FSDPStrategy(activation_checkpointing_policy={torch.nn.Linear})
    fabric = Fabric(devices=1, accelerator="cuda", strategy=strategy)
    fabric.launch()

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointWrapper,
        apply_activation_checkpointing,
    )

    # manually apply activation checkpointing
    apply_activation_checkpointing(model)

    wrappers = {name for name, mod in model.named_modules() if isinstance(mod, CheckpointWrapper)}
    assert wrappers == {"0", "1"}

    # let fabric set up the model, it shouldn't apply activation checkpointing again
    with pytest.warns(match="is configured, but the model already contains checkpointed"):
        model = fabric.setup(model)

    wrappers = {name for name, mod in model._forward_module.named_modules() if isinstance(mod, CheckpointWrapper)}
    assert wrappers == {"_fsdp_wrapped_module.0", "_fsdp_wrapped_module.1"}


@RunIf(min_cuda_gpus=1)
def test_rewrap_warnings():
    from torch.distributed.fsdp import FullyShardedDataParallel
    from torch.distributed.fsdp.wrap import wrap

    strategy = FSDPStrategy(auto_wrap_policy={torch.nn.Linear})
    fabric = Fabric(devices=1, accelerator="cuda", strategy=strategy)
    fabric.launch()
    with fabric.init_module():
        model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.ReLU(), wrap(torch.nn.Linear(1, 1)))
    with pytest.warns(match="the model is already wrapped"):
        model = fabric.setup(model)
    assert not isinstance(model._forward_module, FullyShardedDataParallel)
    assert isinstance(model._forward_module[2], FullyShardedDataParallel)

    if not _TORCH_GREATER_EQUAL_2_1:
        return

    with fabric.init_module(empty_init=True):
        model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.ReLU(), wrap(torch.nn.Linear(1, 1)))
    assert model[0].weight.is_meta
    with pytest.warns(match="there are still parameters on the meta device"):
        fabric_model = fabric.setup(model)
    assert next(fabric_model.parameters()).is_meta


@RunIf(min_cuda_gpus=2, standalone=True)
@pytest.mark.parametrize(
    "precision",
    [
        "32-true",
        pytest.param("16-mixed"),
        pytest.param("bf16-mixed", marks=RunIf(bf16_cuda=True)),
    ],
)
@pytest.mark.parametrize(
    "clip_type",
    [
        pytest.param("norm", marks=pytest.mark.skip("FSDP gradient clipping by norm is not correct.")),
        "val",
    ],
)
def test_clip_gradients(clip_type, precision):
    if clip_type == "norm" and precision == "16-mixed":
        pytest.skip(reason="Clipping by norm with 16-mixed is numerically unstable.")

    strategy = FSDPStrategy(auto_wrap_policy={torch.nn.Linear})
    fabric = Fabric(accelerator="auto", devices=2, precision=precision, strategy=strategy)
    fabric.launch()

    in_features, out_features = 32, 2
    model = torch.nn.Linear(in_features, out_features, bias=False)
    model.weight.data.fill_(0.01)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    model, optimizer = fabric.setup(model, optimizer)

    batch = torch.full((1, in_features), 0.1, device=fabric.device)
    loss = model(batch).sum()

    # The example is constructed such that the gradients are all the same
    fabric.backward(loss)

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    if clip_type == "norm":
        with FSDP.summon_full_params(model._forward_module, with_grads=True):
            norm = torch.linalg.vector_norm(model.weight.grad.detach().cpu(), 2, dtype=torch.float32).item()
        new_norm = norm / 10
        fabric.clip_gradients(model, optimizer, max_norm=new_norm * 10)
        with FSDP.summon_full_params(model._forward_module, with_grads=True):
            assert torch.allclose(
                torch.linalg.vector_norm(model.weight.grad.detach().cpu(), 2, dtype=torch.float32),
                torch.tensor(new_norm),
            )
    elif clip_type == "val":
        val = model.weight.grad[0].item()
        new_val = val / 2.0
        fabric.clip_gradients(model, optimizer, clip_val=new_val)
        assert torch.allclose(model.weight.grad, torch.full_like(model.weight.grad, new_val))
    else:
        raise AssertionError(f"Unknown clip type: {clip_type}")

    optimizer.step()
    optimizer.zero_grad()


@RunIf(min_cuda_gpus=2, standalone=True, min_torch="2.3.0")
def test_save_sharded_and_consolidate_and_load(tmp_path):
    """Test the consolidation of a FSDP-sharded checkpoint into a single file."""

    fabric = Fabric(
        accelerator="cuda",
        strategy=FSDPStrategy(auto_wrap_policy=always_wrap_policy, state_dict_type="sharded"),
        devices=2,
    )
    fabric.launch()

    model = BoringModel()
    optimizer = torch.optim.Adam(model.parameters())
    model, optimizer = fabric.setup(model, optimizer)
    state = {"model": model, "optimizer": optimizer, "steps": 1}

    # run one iteration to init the state of the optimizer
    loss = model(torch.rand(1, 32, device=fabric.device)).sum()
    fabric.backward(loss)
    optimizer.step()

    checkpoint_path_sharded = fabric.broadcast(str(tmp_path / "checkpoint_sharded"))
    fabric.save(checkpoint_path_sharded, state)
    assert set(os.listdir(checkpoint_path_sharded)) == {"meta.pt", ".metadata", "__0_0.distcp", "__1_0.distcp"}

    # consolidate the checkpoint to a single file
    checkpoint_path_full = fabric.broadcast(str(tmp_path / "checkpoint_full.pt"))
    if fabric.global_rank == 0:
        checkpoint = _load_distributed_checkpoint(Path(checkpoint_path_sharded))
        torch.save(checkpoint, checkpoint_path_full)
    fabric.barrier()

    # re-init and load from full checkpoint
    fabric = Fabric(
        accelerator="cuda",
        strategy=FSDPStrategy(auto_wrap_policy=always_wrap_policy),
        devices=2,
    )

    # Hack: we already called launch() on another Fabric instance above
    fabric._launched = True

    model = BoringModel()
    optimizer = torch.optim.Adam(model.parameters())
    model, optimizer = fabric.setup(model, optimizer)
    state = {"model": model, "optimizer": optimizer, "steps": 1}
    fabric.load(checkpoint_path_full, state)


@RunIf(min_cuda_gpus=2, standalone=True)
def test_no_call_to_apply(monkeypatch):
    """Regression test to ensure we're not calling `FSDP.apply()` indirectly (see #19755)."""
    monkeypatch.setattr(torch.distributed.fsdp.FullyShardedDataParallel, "apply", Mock())

    fabric = Fabric(
        accelerator="cuda",
        strategy=FSDPStrategy(auto_wrap_policy=always_wrap_policy),
        devices=2,
    )
    fabric.launch()

    for setup_method in ("setup", "setup_module"):
        model = BoringModel()
        setup = getattr(fabric, setup_method)
        model = setup(model)
        model._forward_module.apply.assert_not_called()
