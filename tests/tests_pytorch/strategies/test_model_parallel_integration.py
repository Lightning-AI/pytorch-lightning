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

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.demos.boring_classes import BoringModel, RandomDataset
from lightning.pytorch.strategies import ModelParallelStrategy
from torch.utils.data import DataLoader, DistributedSampler
from torchmetrics.classification import Accuracy

from tests_pytorch.helpers.runif import RunIf


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

    dp_mesh = device_mesh["data_parallel"]
    assert dp_mesh.ndim == 1  # Hybrid-sharding not supported

    # Fully-shard each layer
    fully_shard(model.w1, mesh=dp_mesh)
    fully_shard(model.w2, mesh=dp_mesh)
    fully_shard(model.w3, mesh=dp_mesh)

    # TODO: Re-enable activation checkpointing
    # Currently, state dict keys get prefixed with '_checkpoint_wrapper' in the keys
    # which leads to mismatches when loading weights into a checkpoint-wrapped module.
    # PyTorch should handle this automatically.

    # model = checkpoint_wrapper(model)

    return model


def _parallelize_feed_forward_fsdp2_tp(model, device_mesh):
    model = _parallelize_feed_forward_tp(model, device_mesh)
    model = _parallelize_feed_forward_fsdp2(model, device_mesh)
    return model


class TemplateModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = FeedForward()

    def training_step(self, batch):
        output = self.model(batch)
        return output.sum()

    def train_dataloader(self):
        dataset_size = 8
        dataset = RandomDataset(32, dataset_size)
        return DataLoader(dataset, batch_size=2)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters())


class FSDP2Model(TemplateModel):
    def configure_model(self):
        _parallelize_feed_forward_fsdp2(self.model, device_mesh=self.device_mesh)


class TensorParallelModel(TemplateModel):
    def configure_model(self):
        _parallelize_feed_forward_tp(self.model, device_mesh=self.device_mesh)


class FSDP2TensorParallelModel(TemplateModel):
    def configure_model(self):
        _parallelize_feed_forward_fsdp2_tp(self.model, device_mesh=self.device_mesh)


@RunIf(min_torch="2.3", standalone=True, min_cuda_gpus=4)
def test_setup_device_mesh():
    from torch.distributed.device_mesh import DeviceMesh

    for dp_size, tp_size in ((1, 4), (4, 1), (2, 2)):
        strategy = ModelParallelStrategy(
            data_parallel_size=dp_size,
            tensor_parallel_size=tp_size,
        )
        trainer = Trainer(
            accelerator="auto",
            devices=4,
            strategy=strategy,
            logger=False,
            enable_checkpointing=False,
            max_steps=1,
        )

        class Model(BoringModel):
            def configure_model(self):
                device_mesh = self.device_mesh
                assert isinstance(device_mesh, DeviceMesh)
                assert device_mesh.device_type == model.device.type
                assert device_mesh.mesh_dim_names == ("data_parallel", "tensor_parallel")
                assert device_mesh.size(0) == dp_size
                assert device_mesh.size(1) == tp_size
                assert device_mesh.ndim == 2

        model = Model()
        trainer.fit(model)

    # Passing "auto" will select internode and intranode dimensions automatically
    strategy = ModelParallelStrategy(
        data_parallel_size="auto",
        tensor_parallel_size="auto",
    )
    trainer = Trainer(
        accelerator="auto",
        devices=4,
        num_nodes=1,
        strategy=strategy,
        logger=False,
        enable_checkpointing=False,
        max_steps=1,
    )

    class Model(BoringModel):
        def configure_model(self):
            device_mesh = self.device_mesh
            assert device_mesh.mesh_dim_names == ("data_parallel", "tensor_parallel")
            assert device_mesh.size(0) == 1
            assert device_mesh.size(1) == 4

    model = Model()
    trainer.fit(model)


@RunIf(min_torch="2.3", standalone=True, min_cuda_gpus=2)
def test_tensor_parallel():
    from torch.distributed._tensor import DTensor

    class Model(TensorParallelModel):
        def on_train_start(self):
            device_mesh = self.device_mesh
            optimizer = self.optimizers()
            assert all(
                tensor.device_mesh == device_mesh["tensor_parallel"] for tensor in optimizer.param_groups[0]["params"]
            )
            assert all(isinstance(weight, DTensor) for weight in self.model.parameters())
            assert self.model.w1.weight.device_mesh == device_mesh["tensor_parallel"]

            # No data sharding, all GPUs get the same input inside a TP group
            dataloader = self.trainer.train_dataloader
            assert len(dataloader) == 8 // dataloader.batch_size
            assert isinstance(dataloader.sampler, DistributedSampler)

        def training_step(self, batch):
            # All batches must be identical across TP group
            batches = self.all_gather(batch)
            assert all(torch.equal(batches[0], batches[i]) for i in range(1, len(batches)))
            return super().training_step(batch)

    trainer = Trainer(
        accelerator="auto",
        devices=2,
        strategy=ModelParallelStrategy(),
        max_steps=2,
        enable_checkpointing=False,
        logger=False,
    )

    seed_everything(0)
    with trainer.init_module(empty_init=True):
        model = Model()

    trainer.fit(model)


@RunIf(min_torch="2.3", standalone=True, min_cuda_gpus=4)
def test_fsdp2_tensor_parallel():
    from torch.distributed._tensor import DTensor

    class Model(FSDP2TensorParallelModel):
        def on_train_start(self):
            optimizer = self.optimizers()
            assert all(isinstance(weight, DTensor) for weight in self.model.parameters())
            assert all(isinstance(tensor, DTensor) for tensor in optimizer.param_groups[0]["params"])
            assert self.model.w1.weight.device_mesh.ndim == 2
            assert self.model.w1.weight.device_mesh.size(0) == 2
            assert self.model.w1.weight.device_mesh.size(1) == 2
            assert all(weight.device.type != "meta" for weight in self.model.parameters())
            assert all(tensor.device_mesh.ndim == 2 for tensor in optimizer.param_groups[0]["params"])
            assert all(tensor.device.type != "meta" for tensor in optimizer.param_groups[0]["params"])

            # No data sharding across TP dimension, sharding across data-parallel dimension only
            device_mesh = self.device_mesh
            dp_mesh = device_mesh["data_parallel"]
            dataloader = self.trainer.train_dataloader
            assert len(dataloader) == 8 // dataloader.batch_size // dp_mesh.size()
            assert isinstance(dataloader.sampler, DistributedSampler)

        def training_step(self, batch):
            batches = self.all_gather(batch)
            dp_mesh = self.device_mesh["data_parallel"]
            tp_mesh = self.device_mesh["tensor_parallel"]

            # Batches across the TP dimension must be identical
            batches_tp = batches[tp_mesh.mesh]
            assert all(torch.equal(batches_tp[0], batches_tp[i]) for i in range(1, len(batches_tp)))
            # Batches across the DP dimension must be different
            batches_dp = batches[dp_mesh.mesh]
            assert all(not torch.equal(batches_dp[0], batches_dp[i]) for i in range(1, len(batches_dp)))

            return super().training_step(batch)

    strategy = ModelParallelStrategy(
        data_parallel_size=2,
        tensor_parallel_size=2,
    )
    trainer = Trainer(
        accelerator="auto",
        devices=4,
        strategy=strategy,
        max_steps=2,
        enable_checkpointing=False,
        logger=False,
    )

    seed_everything(0)
    with trainer.init_module(empty_init=True):
        model = Model()

    trainer.fit(model)


@RunIf(min_torch="2.3", min_cuda_gpus=2, standalone=True)
def test_modules_without_parameters(tmp_path):
    """Test that TorchMetrics get moved to the device despite not having any parameters."""

    class MetricsModel(TensorParallelModel):
        def __init__(self):
            super().__init__()
            self.metric = Accuracy("multiclass", num_classes=10)
            assert self.metric.device == self.metric.tp.device == torch.device("cpu")

        def setup(self, stage) -> None:
            assert self.metric.device == self.metric.tp.device == torch.device("cpu")

        def training_step(self, batch):
            assert self.metric.device.type == self.metric.tp.device.type == "cuda"
            self.metric(torch.rand(2, 10, device=self.device), torch.randint(0, 10, size=(2,), device=self.device))
            return super().training_step(batch)

    model = MetricsModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        accelerator="cuda",
        devices=2,
        strategy=ModelParallelStrategy(),
        max_steps=1,
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(model)


@RunIf(min_torch="2.3", min_cuda_gpus=2, standalone=True)
@pytest.mark.parametrize(
    ("precision", "expected_dtype"),
    [
        ("32-true", torch.float32),
        ("16-true", torch.float16),
        pytest.param("bf16-true", torch.bfloat16, marks=RunIf(bf16_cuda=True)),
    ],
)
def test_module_init_context(precision, expected_dtype, tmp_path):
    """Test that the module under the init-context gets moved to the right device and dtype."""

    class Model(FSDP2Model):
        def on_train_start(self):
            assert self.model.w1.weight.device == torch.device("cuda", self.local_rank)
            assert self.model.w1.weight.dtype == expected_dtype
            optimizer = self.optimizers(use_pl_optimizer=False)
            assert optimizer.param_groups[0]["params"][0].device.type == "cuda"

    def _run_setup_assertions(empty_init, expected_device):
        trainer = Trainer(
            default_root_dir=tmp_path,
            accelerator="cuda",
            devices=2,
            strategy=ModelParallelStrategy(),
            precision=precision,
            max_steps=1,
            barebones=True,
            enable_checkpointing=False,
            logger=False,
        )
        with trainer.init_module(empty_init=empty_init):
            model = Model()

        # The model is on the CPU/meta-device until after `ModelParallelStrategy.setup()`
        assert model.model.w1.weight.device == expected_device
        assert model.model.w1.weight.dtype == expected_dtype
        trainer.fit(model)

    # Case 1: No empty init
    _run_setup_assertions(empty_init=False, expected_device=torch.device("cpu"))

    # Case 2: Empty-init with PyTorch >= 2.1 supports meta device
    _run_setup_assertions(empty_init=True, expected_device=torch.device("meta"))


@RunIf(min_torch="2.3", min_cuda_gpus=2, skip_windows=True, standalone=True)
@pytest.mark.parametrize("save_distributed_checkpoint", [True, False])
def test_strategy_state_dict(tmp_path, save_distributed_checkpoint):
    """Test that the strategy returns the correct state dict of the LightningModule."""
    model = FSDP2Model()
    correct_state_dict = model.state_dict()  # State dict before wrapping

    strategy = ModelParallelStrategy(save_distributed_checkpoint=save_distributed_checkpoint)
    trainer = Trainer(
        default_root_dir=tmp_path,
        accelerator="cuda",
        devices=2,
        strategy=strategy,
        max_epochs=1,
        barebones=True,
    )
    trainer.fit(model)

    state_dict = trainer.strategy.lightning_module_state_dict()

    if save_distributed_checkpoint:
        # All ranks return a state dict
        assert len(state_dict) > 0
        # State dict should contain same keys as non-distributed state dict
        assert list(state_dict.keys()) == list(correct_state_dict.keys())
    else:
        if trainer.global_rank != 0:
            # The full state-dict is only returned on rank 0
            assert len(state_dict) == 0
            return
        # State dict should contain same keys as non-distributed state dict
        assert list(state_dict.keys()) == list(correct_state_dict.keys())


@RunIf(min_torch="2.3", min_cuda_gpus=2, skip_windows=True, standalone=True)
def test_load_full_state_checkpoint_into_regular_model(tmp_path):
    """Test that a full-state checkpoint saved from a distributed model can be loaded back into a regular model."""

    # Save a regular full-state checkpoint from a distributed model
    model = FSDP2Model()
    strategy = ModelParallelStrategy(save_distributed_checkpoint=False)
    trainer = Trainer(
        default_root_dir=tmp_path,
        accelerator="gpu",
        devices=2,
        strategy=strategy,
        max_epochs=1,
        barebones=True,
    )
    trainer.fit(model)
    model_path = tmp_path / "last.ckpt"
    model_path = trainer.strategy.broadcast(model_path)
    trainer.save_checkpoint(model_path)
    model_state_dict = trainer.strategy.lightning_module_state_dict()
    optimizer_state_dict = trainer.strategy.optimizer_state(model.optimizers())

    if trainer.global_rank != 0:
        assert len(model_state_dict) == 0
        assert len(optimizer_state_dict) == 0

    # Create a regular model and load the checkpoint into it
    model = TemplateModel()
    trainer = Trainer(default_root_dir=tmp_path, accelerator="gpu", devices=2, strategy="ddp", max_epochs=1)
    trainer.fit(model, ckpt_path=model_path)
    restored_model_state_dict = trainer.strategy.lightning_module_state_dict()
    restored_optimizer_state_dict = trainer.strategy.optimizer_state(model.optimizers())

    if trainer.global_rank == 0:
        assert len(model_state_dict) == len(restored_model_state_dict)
        assert len(optimizer_state_dict) == len(restored_optimizer_state_dict)
        torch.testing.assert_close(model_state_dict, restored_model_state_dict, atol=0, rtol=0)
        torch.testing.assert_close(optimizer_state_dict, restored_optimizer_state_dict, atol=0, rtol=0)
    trainer.strategy.barrier()


@RunIf(min_torch="2.4", min_cuda_gpus=2, skip_windows=True, standalone=True)
def test_load_standard_checkpoint_into_distributed_model(tmp_path):
    """Test that a regular checkpoint (weights and optimizer states) can be loaded into a distributed model."""

    # Save a regular DDP checkpoint
    model = TemplateModel()
    trainer = Trainer(default_root_dir=tmp_path, accelerator="gpu", devices=2, strategy="ddp", max_epochs=1)
    trainer.fit(model)
    model_path = tmp_path / "last.ckpt"
    model_path = trainer.strategy.broadcast(model_path)
    trainer.save_checkpoint(model_path)
    model_state_dict = trainer.strategy.lightning_module_state_dict()
    optimizer_state_dict = trainer.strategy.optimizer_state(model.optimizers())

    # Create a distributed model and load the checkpoint into it
    model = FSDP2Model()
    strategy = ModelParallelStrategy(save_distributed_checkpoint=False)
    trainer = Trainer(
        default_root_dir=tmp_path,
        accelerator="gpu",
        devices=2,
        strategy=strategy,
        max_epochs=1,
        barebones=True,
    )
    trainer.fit(model, ckpt_path=model_path)
    restored_model_state_dict = trainer.strategy.lightning_module_state_dict()
    restored_optimizer_state_dict = trainer.strategy.optimizer_state(model.optimizers())

    if trainer.global_rank != 0:
        assert len(restored_model_state_dict) == 0
        assert len(restored_optimizer_state_dict) == 0
    if trainer.global_rank == 0:
        assert len(model_state_dict) == len(restored_model_state_dict)
        assert len(optimizer_state_dict) == len(restored_optimizer_state_dict)
        torch.testing.assert_close(model_state_dict, restored_model_state_dict, atol=0, rtol=0)
        torch.testing.assert_close(optimizer_state_dict, restored_optimizer_state_dict, atol=0, rtol=0)
    trainer.strategy.barrier()


@RunIf(min_torch="2.4", min_cuda_gpus=2, standalone=True)
def test_save_load_sharded_state_dict(tmp_path):
    """Test saving and loading with the distributed state dict format."""

    class CheckpointModel(FSDP2Model):
        def __init__(self, params_to_compare=None):
            super().__init__()
            self.params_to_compare = params_to_compare

        def on_train_start(self):
            if self.params_to_compare is None:
                return
            for p0, p1 in zip(self.params_to_compare, self.trainer.model.parameters()):
                assert torch.equal(p0, p1.full_tensor())

    seed_everything(0)

    strategy = ModelParallelStrategy(save_distributed_checkpoint=True)
    trainer_kwargs = {
        "default_root_dir": tmp_path,
        "accelerator": "cuda",
        "devices": 2,
        "max_epochs": 1,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "logger": False,
    }

    # Initial training
    model = CheckpointModel()
    trainer = Trainer(**trainer_kwargs, strategy=strategy)
    trainer.fit(model)
    params_before = [p.full_tensor() for p in trainer.model.parameters()]

    checkpoint_path = Path(trainer.strategy.broadcast(trainer.checkpoint_callback.best_model_path))
    assert set(os.listdir(checkpoint_path)) == {"meta.pt", ".metadata", "__0_0.distcp", "__1_0.distcp"}

    metadata = torch.load(checkpoint_path / "meta.pt")
    assert "pytorch-lightning_version" in metadata
    assert len(metadata["callbacks"]) == 1  # model checkpoint callback
    assert "state_dict" not in metadata
    assert "optimizer_states" not in metadata

    # Load checkpoint and continue training
    trainer_kwargs.update(max_epochs=2)
    model = CheckpointModel(params_to_compare=params_before)
    strategy = ModelParallelStrategy(save_distributed_checkpoint=True)
    trainer = Trainer(**trainer_kwargs, strategy=strategy)
    trainer.fit(model, ckpt_path=checkpoint_path)
