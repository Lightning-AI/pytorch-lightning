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
from pathlib import Path
from unittest import mock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.fabric import Fabric
from lightning.fabric.strategies import ModelParallelStrategy
from lightning.fabric.utilities.load import _load_distributed_checkpoint
from torch.utils.data import DataLoader, DistributedSampler

from tests_fabric.helpers.datasets import RandomDataset
from tests_fabric.helpers.runif import RunIf


@RunIf(min_torch="2.3", standalone=True, min_cuda_gpus=4)
def test_setup_device_mesh():
    from torch.distributed.device_mesh import DeviceMesh

    for dp_size, tp_size in ((1, 4), (4, 1), (2, 2)):
        strategy = ModelParallelStrategy(
            parallelize_fn=(lambda m, _: m),
            data_parallel_size=dp_size,
            tensor_parallel_size=tp_size,
        )
        fabric = Fabric(accelerator="auto", devices=4, strategy=strategy)
        fabric.launch()

        device_mesh = fabric.strategy.device_mesh
        assert isinstance(device_mesh, DeviceMesh)
        assert device_mesh.device_type == fabric.device.type
        assert device_mesh.mesh_dim_names == ("data_parallel", "tensor_parallel")
        assert device_mesh.size(0) == dp_size
        assert device_mesh.size(1) == tp_size
        assert device_mesh.ndim == 2

        fabric.barrier()

    # Passing sizes that don't multiply to the world size raises an error
    for invalid_dp_size, invalid_tp_size in ((1, 1), (2, 3), (4, 2)):
        strategy = ModelParallelStrategy(
            parallelize_fn=(lambda m, _: m),
            data_parallel_size=invalid_dp_size,
            tensor_parallel_size=invalid_tp_size,
        )
        fabric = Fabric(accelerator="auto", devices=4, strategy=strategy)
        with pytest.raises(RuntimeError, match="multiplied should equal the world size"):
            fabric.launch()

        fabric.barrier()

    # Passing "auto" will select internode and intranode dimensions automatically
    strategy = ModelParallelStrategy(
        parallelize_fn=(lambda m, _: m),
        data_parallel_size="auto",
        tensor_parallel_size="auto",
    )
    fabric = Fabric(accelerator="auto", devices=4, num_nodes=1, strategy=strategy)
    fabric.launch()
    assert fabric.strategy.device_mesh.mesh_dim_names == ("data_parallel", "tensor_parallel")
    assert fabric.strategy.device_mesh.size(0) == 1
    assert fabric.strategy.device_mesh.size(1) == 4


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(32, 64)
        self.w2 = nn.Linear(32, 64)
        self.w3 = nn.Linear(64, 32)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


def _parallelize_feed_forward_tp(model, device_mesh):
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

    tp_mesh = device_mesh["tensor_parallel"]
    tp_plan = {
        "w1": ColwiseParallel(),
        "w2": ColwiseParallel(),
        "w3": RowwiseParallel(),
    }
    parallelize_module(model, tp_mesh, tp_plan)
    return model


def _parallelize_feed_forward_fsdp2(model, device_mesh):
    from torch.distributed._composable.fsdp.fully_shard import fully_shard
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    dp_mesh = device_mesh["data_parallel"]
    assert dp_mesh.ndim == 1  # Hybrid-sharding not supported

    # Fully-shard each layer
    fully_shard(model.w1, mesh=dp_mesh)
    fully_shard(model.w2, mesh=dp_mesh)
    fully_shard(model.w3, mesh=dp_mesh)

    # Activation checkpointing
    model = checkpoint_wrapper(model)

    return model


def _parallelize_feed_forward_fsdp2_tp(model, device_mesh):
    model = _parallelize_feed_forward_tp(model, device_mesh)
    model = _parallelize_feed_forward_fsdp2(model, device_mesh)
    return model


@RunIf(min_torch="2.3", standalone=True, min_cuda_gpus=2)
def test_tensor_parallel():
    from torch.distributed._tensor import DTensor

    strategy = ModelParallelStrategy(parallelize_fn=_parallelize_feed_forward_tp)
    fabric = Fabric(accelerator="auto", devices=2, strategy=strategy)
    fabric.launch()

    fabric.seed_everything(0)

    with fabric.init_module(empty_init=True):
        model = FeedForward()

    optimizer = torch.optim.AdamW(model.parameters())
    model, optimizer = fabric.setup(model, optimizer)

    assert all(isinstance(weight, DTensor) for weight in model.parameters())
    assert model.w1.weight.device_mesh == fabric.strategy.device_mesh["tensor_parallel"]

    dataset_size = 6
    dataset = RandomDataset(32, dataset_size)
    dataloader = DataLoader(dataset, batch_size=2)
    dataloader = fabric.setup_dataloaders(dataloader)

    # No data sharding, all GPUs get the same input inside a TP group
    assert len(dataloader) == dataset_size // dataloader.batch_size
    assert isinstance(dataloader.sampler, DistributedSampler)

    for _, batch in enumerate(dataloader):
        # All batches must be identical across TP group
        batches = fabric.all_gather(batch)
        assert all(torch.equal(batches[0], batches[i]) for i in range(1, len(batches)))

        output = model(batch)
        fabric.backward(output.sum())
        optimizer.step()
        optimizer.zero_grad()


@RunIf(min_torch="2.3", standalone=True, min_cuda_gpus=4)
def test_fsdp2_tensor_parallel():
    from torch.distributed._tensor import DTensor

    strategy = ModelParallelStrategy(
        parallelize_fn=_parallelize_feed_forward_fsdp2_tp,
        data_parallel_size=2,
        tensor_parallel_size=2,
    )
    fabric = Fabric(accelerator="auto", devices=4, strategy=strategy)
    fabric.launch()

    fabric.seed_everything(0)

    with fabric.init_module(empty_init=True):
        model = FeedForward()

    optimizer = torch.optim.AdamW(model.parameters())
    model, optimizer = fabric.setup(model, optimizer)

    assert all(isinstance(weight, DTensor) for weight in model.parameters())
    assert model.w1.weight.device_mesh.ndim == 2
    assert model.w1.weight.device_mesh.size(0) == 2
    assert model.w1.weight.device_mesh.size(1) == 2
    assert all(weight.device.type != "meta" for weight in model.parameters())

    dataset_size = 8
    dataset = RandomDataset(32, dataset_size)
    dataloader = DataLoader(dataset, batch_size=2)
    dataloader = fabric.setup_dataloaders(dataloader)

    # No data sharding across TP dimension, sharding across data-parallel dimension only
    dp_mesh = fabric.strategy.device_mesh["data_parallel"]
    tp_mesh = fabric.strategy.device_mesh["tensor_parallel"]
    assert len(dataloader) == dataset_size // dataloader.batch_size // dp_mesh.size()
    assert isinstance(dataloader.sampler, DistributedSampler)

    for _, batch in enumerate(dataloader):
        batches = fabric.all_gather(batch)
        # Batches across the TP dimension must be identical
        batches_tp = batches[tp_mesh.mesh]
        assert all(torch.equal(batches_tp[0], batches_tp[i]) for i in range(1, len(batches_tp)))
        # Batches across the DP dimension must be different
        batches_dp = batches[dp_mesh.mesh]
        assert all(not torch.equal(batches_dp[0], batches_dp[i]) for i in range(1, len(batches_dp)))

        output = model(batch)
        fabric.backward(output.sum())
        optimizer.step()
        optimizer.zero_grad()


@RunIf(min_torch="2.3", min_cuda_gpus=4, standalone=True)
@pytest.mark.parametrize(
    "precision",
    [
        pytest.param(
            "16-mixed", marks=pytest.mark.xfail(reason="Precision plugin does not implement ShardedGradScaler yet")
        ),
        pytest.param("bf16-mixed", marks=RunIf(bf16_cuda=True)),
    ],
)
def test_train_save_load(precision, tmp_path):
    """Test 2D-parallel training, saving and loading precision settings."""
    strategy = ModelParallelStrategy(
        _parallelize_feed_forward_fsdp2_tp,
        data_parallel_size=2,
        tensor_parallel_size=2,
    )
    fabric = Fabric(accelerator="cuda", devices=4, strategy=strategy, precision=precision)
    fabric.launch()

    fabric.seed_everything(0)
    with fabric.init_module(empty_init=True):
        model = FeedForward()
    model = fabric.setup(model)
    optimizer = torch.optim.AdamW(model.parameters())
    optimizer = fabric.setup_optimizers(optimizer)
    output = model(torch.rand(2, 32, device=fabric.device))
    fabric.backward(output.sum())
    optimizer.step()
    optimizer.zero_grad()

    checkpoint_path = fabric.broadcast(str(tmp_path / "dist-checkpoint"))

    params_before = [p.full_tensor().clone() for p in model.parameters()]
    state = {"model": model, "optimizer": optimizer, "steps": 1}
    fabric.save(checkpoint_path, state)
    assert set(os.listdir(checkpoint_path)) == {
        ".metadata",
        "__0_0.distcp",
        "__1_0.distcp",
        "__2_0.distcp",
        "__3_0.distcp",
    }

    # re-init all objects and resume
    strategy = ModelParallelStrategy(
        _parallelize_feed_forward_fsdp2_tp,
        data_parallel_size=2,
        tensor_parallel_size=2,
    )
    fabric = Fabric(accelerator="cuda", devices=4, strategy=strategy, precision=precision)
    fabric.launch()

    fabric.seed_everything(0)
    with fabric.init_module(empty_init=True):
        model = FeedForward()
    model = fabric.setup(model)
    optimizer = torch.optim.AdamW(model.parameters())
    optimizer = fabric.setup_optimizers(optimizer)
    output = model(torch.rand(2, 32, device=fabric.device))
    fabric.backward(output.sum())
    optimizer.step()
    optimizer.zero_grad()

    # check correctness with loaded state
    state = {"model": model, "optimizer": optimizer, "steps": 0}
    metadata = fabric.load(checkpoint_path, state)
    for p0, p1 in zip(params_before, (p.full_tensor() for p in model.parameters())):
        torch.testing.assert_close(p0, p1, atol=0, rtol=0, equal_nan=True)

    # check user data in state reloaded
    # TODO: This should be 1, torch.distributed.checkpoint only supports tensor data
    assert state["steps"] == 0
    assert not metadata

    # TODO: Test strict and non-strict loading here once supported
    # attempt to load a key not in the metadata checkpoint
    # state = {"model": model, "coconut": 11}
    # with pytest.raises(KeyError, match="The requested state contains a key 'coconut' that does not exist"):
    #     fabric.load(checkpoint_path, state)

    # # `strict=False` ignores the missing key
    # state = {"model": trainer.model, "coconut": 11}
    # fabric.load(checkpoint_path, state, strict=False)
    # assert state["coconut"] == 11


@RunIf(min_torch="2.3", min_cuda_gpus=2, skip_windows=True, standalone=True)
@pytest.mark.parametrize("move_to_device", [True, False])
@mock.patch("lightning.fabric.wrappers._FabricModule")
def test_setup_module_move_to_device(fabric_module_mock, move_to_device):
    """Test that `move_to_device` does nothing, ModelParallel decides which device parameters get moved to which device
    (sharding)."""
    from torch.distributed._tensor import DTensor

    strategy = ModelParallelStrategy(parallelize_fn=_parallelize_feed_forward_fsdp2)
    fabric = Fabric(accelerator="cuda", devices=2, strategy=strategy)
    fabric.launch()

    model = FeedForward()
    fabric_model = fabric.setup_module(model, move_to_device=move_to_device)
    fabric_module_mock.assert_not_called()

    # the linear layer got sharded and each part is on the expected device
    assert fabric_model.w1.weight.device == torch.device("cuda", fabric.local_rank)
    assert isinstance(fabric_model.w1.weight, DTensor)

    # The _DeviceDtypeModuleMixin currently can't represent the device in a meaningful way for models with pieces on
    # different devices
    assert fabric_model.device == torch.device("cuda", fabric.local_rank)
    assert fabric.device == torch.device("cuda", fabric.local_rank)


@RunIf(min_torch="2.3", min_cuda_gpus=2, skip_windows=True, standalone=True)
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
    strategy = ModelParallelStrategy(parallelize_fn=_parallelize_feed_forward_fsdp2)
    fabric = Fabric(accelerator="cuda", devices=2, strategy=strategy, precision=precision)
    fabric.launch()

    def _run_setup_assertions(empty_init, expected_device):
        with fabric.init_module(empty_init=empty_init):
            model = FeedForward()

        # The model is on the CPU/meta-device until after `.setup()``
        assert all(weight.device == expected_device for weight in model.parameters())
        assert all(weight.dtype == expected_dtype for weight in model.parameters())
        model = fabric.setup(model)
        # Parameters get sharded in `.setup()` and moved to the target device
        assert all(weight.device == torch.device("cuda", fabric.local_rank) for weight in model.parameters())
        assert all(weight.dtype == expected_dtype for weight in model.parameters())

    _run_setup_assertions(empty_init=False, expected_device=torch.device("cpu"))
    _run_setup_assertions(empty_init=True, expected_device=torch.device("meta"))


def _parallelize_single_linear_tp_fsdp2(model, device_mesh):
    from torch.distributed._composable.fsdp.fully_shard import fully_shard
    from torch.distributed.tensor.parallel import ColwiseParallel, parallelize_module

    dp_mesh = device_mesh["data_parallel"]
    tp_mesh = device_mesh["tensor_parallel"]

    parallelize_module(model, tp_mesh, ColwiseParallel())
    fully_shard(model, mesh=dp_mesh)
    return model


@RunIf(min_torch="2.3", min_cuda_gpus=2, standalone=True)
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
        pytest.param("norm", marks=pytest.mark.skip("Gradient clipping by norm is not correct.")),
        pytest.param(
            "val",
            marks=pytest.mark.xfail(raises=RuntimeError, reason="Clipping DTensor by value raises error in PyTorch"),
        ),
    ],
)
def test_clip_gradients(clip_type, precision):
    if clip_type == "norm" and precision == "16-mixed":
        pytest.skip(reason="Clipping by norm with 16-mixed is numerically unstable.")

    strategy = ModelParallelStrategy(_parallelize_single_linear_tp_fsdp2)
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

    if clip_type == "norm":
        norm = torch.linalg.vector_norm(model.weight.grad.full_tensor().detach().cpu(), 2, dtype=torch.float32).item()
        new_norm = norm / 10
        fabric.clip_gradients(model, optimizer, max_norm=new_norm * 10)
        assert torch.allclose(
            torch.linalg.vector_norm(model.weight.grad.full_tensor().detach().cpu(), 2, dtype=torch.float32),
            torch.tensor(new_norm),
        )
    elif clip_type == "val":
        val = model.weight.grad.full_tensor()[0, 0].item()
        new_val = val / 2.0
        fabric.clip_gradients(model, optimizer, clip_val=new_val)
        assert torch.allclose(model.weight.full_tensor().grad, torch.full_like(model.weight.grad, new_val))
    else:
        raise AssertionError(f"Unknown clip type: {clip_type}")

    optimizer.step()
    optimizer.zero_grad()


# TODO: Support loading full checkpoint
@pytest.mark.xfail(raises=NotADirectoryError, reason="Loading from full checkpoint not supported yet.")
@RunIf(min_torch="2.3", min_cuda_gpus=4, standalone=True)
def test_save_sharded_and_consolidate_and_load(tmp_path):
    """Test the consolidation of a distributed (DTensor) checkpoint into a single file."""
    strategy = ModelParallelStrategy(
        _parallelize_feed_forward_fsdp2_tp,
        data_parallel_size=2,
        tensor_parallel_size=2,
    )
    fabric = Fabric(accelerator="cuda", devices=4, strategy=strategy)
    fabric.launch()

    model = FeedForward()
    optimizer = torch.optim.Adam(model.parameters())
    model, optimizer = fabric.setup(model, optimizer)
    state = {"model": model, "optimizer": optimizer, "steps": 1}

    # run one iteration to init the state of the optimizer
    loss = model(torch.rand(1, 32, device=fabric.device)).sum()
    fabric.backward(loss)
    optimizer.step()

    checkpoint_path_sharded = fabric.broadcast(str(tmp_path / "checkpoint_sharded"))
    fabric.save(checkpoint_path_sharded, state)
    assert set(os.listdir(checkpoint_path_sharded)) == {
        ".metadata",
        "__0_0.distcp",
        "__1_0.distcp",
        "__2_0.distcp",
        "__3_0.distcp",
    }

    # consolidate the checkpoint to a single file
    checkpoint_path_full = fabric.broadcast(str(tmp_path / "checkpoint_full.pt"))
    if fabric.global_rank == 0:
        checkpoint = _load_distributed_checkpoint(Path(checkpoint_path_sharded))
        torch.save(checkpoint, checkpoint_path_full)
    fabric.barrier()

    # re-init and load from full checkpoint
    strategy = ModelParallelStrategy(_parallelize_feed_forward_fsdp2_tp)
    fabric = Fabric(accelerator="cuda", devices=4, strategy=strategy)
    fabric.launch()

    import time

    print(checkpoint_path_full)
    time.sleep(60)

    model = FeedForward()
    optimizer = torch.optim.Adam(model.parameters())
    model, optimizer = fabric.setup(model, optimizer)
    state = {"model": model, "optimizer": optimizer, "steps": 1}
    fabric.load(checkpoint_path_full, state)
