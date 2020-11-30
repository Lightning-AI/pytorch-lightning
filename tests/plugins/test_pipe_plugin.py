import os
import platform
import sys
import time
from copy import deepcopy
from unittest import mock

import numpy as np
import pytest
import torch
import torch.distributed as torch_distrib
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.plugins.pipe_plugin import HAS_FAIRSCALE, PipePlugin

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

from tests.backends.launcher import DDPLauncher  # noqa E402
from tests.base.boring_model import RandomDataset  # noqa E402


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.skipif(platform.system() == "Windows",
                    reason="Distributed training is not supported on Windows")
@pytest.mark.skipif(not HAS_FAIRSCALE, reason="Fairscale is not available")
@DDPLauncher.run("--accelerator ddp --gpus 2")
def test_ddp_pipe_plugin_correctness_multi_gpu(tmpdir, args=None):
    run_pipe_correctness(gpus=args.gpus, accelerator=args.accelerator)


class PipeRandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)
        self.len *= 4
        self.data = self.data.repeat((4, 1))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class PipeBoringModel(LightningModule):

    count = 0

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(torch.nn.Linear(32, 4 * 2048, bias=False), nn.ReLU(), nn.Linear(4 * 2048, 2, bias=False))
        self._example_input_array = torch.randn((1, 32))

    def forward(self, x):
        return self.layers(x)

    def loss(self, prediction):
        # An arbitrary loss to have a loss that updates the model weights during `Trainer.fit` calls
        return torch.nn.functional.mse_loss(prediction, torch.ones_like(prediction))

    def training_step_normal(self, batch, batch_idx):
        # manual
        opt = self.optimizers()
        output = self.layers(batch)
        print(output)
        loss = self.loss(output)
        self.manual_backward(loss, opt)
        self.manual_optimizer_step(opt)
        self.count += 1

    def training_step_pipe(self, batch, batch_idx):
        opt = self.optimizers()
        output = self.layers(batch)
        loss = torch.zeros(1).to(self.device)

        if self.final_stage:
            print(output)
            loss = self.loss(output)
            self.manual_backward(loss, opt)
            assert torch.stack([torch.abs(p.grad).sum() for p in self.parameters()]).sum() > 0
            self.manual_optimizer_step(opt)
            assert torch.stack([torch.abs(p.grad).sum() for p in self.parameters()]).sum() == 0
        else:
            self.back_helper(output)
            self.manual_optimizer_step(opt)
        self.count += 1

    def validation_step_normal(self, batch, batch_idx):
        output = self.layers(batch)
        loss = self.loss(output)

    def validation_step_pipe(self, batch, batch_idx):
        output = self.layers(batch)
        if self.final_stage:
            loss = self.loss(output)

    def training_step_end(self, training_step_outputs):
        return training_step_outputs

    def train_dataloader_normal_2(self):
        seed_everything(42)
        return torch.utils.data.DataLoader(PipeRandomDataset(32, 1), batch_size=2)

    def val_dataloader_normal_2(self):
        return torch.utils.data.DataLoader(PipeRandomDataset(32, 1), batch_size=2)

    def train_dataloader_normal(self):
        seed_everything(42)
        return torch.utils.data.DataLoader(RandomDataset(32, 1), batch_size=1)

    def val_dataloader_normal(self):
        return torch.utils.data.DataLoader(RandomDataset(32, 1), batch_size=1)

    def train_dataloader_pipe(self):
        seed_everything(42)
        return torch.utils.data.DataLoader(PipeRandomDataset(32, 1), batch_size=2)

    def val_dataloader_pipe(self):
        return torch.utils.data.DataLoader(PipeRandomDataset(32, 1), batch_size=2)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.layers.parameters(), lr=0.1)
        return optimizer

    @property
    def automatic_optimization(self) -> bool:
        return False


def record_ddp_fit_model_stats(trainer, model, gpus):
    """
    Helper to calculate wall clock time for fit + max allocated memory.
    Args:
        trainer: The trainer object.
        model: The LightningModule.
        gpus: Number of GPUs in test.
    Returns:
        Max Memory if using GPUs, and total wall clock time.
    """
    max_memory = None

    time_start = time.perf_counter()
    if gpus > 0:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    trainer.fit(model)

    if gpus > 0:
        torch.cuda.synchronize()
        max_memory = torch.cuda.max_memory_allocated() / 2 ** 20

    total_time = time.perf_counter() - time_start

    return max_memory, total_time


def run_pipe_correctness(
        accelerator='ddp_spawn',
        gpus=0,
        precision=32,
        max_percent_speed_diff=0.25,
        model_cls=PipeBoringModel):
    """
    Ensures that the trained model is identical to the standard DDP implementation.
    Also checks for speed/memory regressions, we should expect always less memory but performance to fluctuate.
    Args:
        accelerator: Accelerator type for test.
        gpus: Number of GPUS to enable.
        precision: Whether to use AMP or normal FP32 training.
        max_percent_speed_diff: The maximum speed difference compared to normal DDP training.
        This is more a safety net for variability in CI which can vary in speed, not for benchmarking.
        model_cls: Model class to use for test.
    """
    # Train normal DDP with batch_size 2
    seed_everything(42)
    ddp_model = model_cls()
    ddp_model.training_step = ddp_model.training_step_normal
    ddp_model.validation_step = ddp_model.validation_step_normal
    ddp_model.train_dataloader = ddp_model.train_dataloader_normal_2
    ddp_model.val_dataloader = ddp_model.train_dataloader_normal_2
    ddp_model_clone = deepcopy(ddp_model)

    trainer = Trainer(
        fast_dev_run=True,
        max_epochs=1,
        gpus=gpus,
        precision=precision,
        accelerator=accelerator,
    )

    max_ddp_memory_2, ddp_time_2 = record_ddp_fit_model_stats(
        trainer=trainer,
        model=ddp_model,
        gpus=gpus
    )

    torch.cuda.empty_cache()
    torch_distrib.barrier()

    # Train normal DDP with batch_size 1
    seed_everything(42)
    ddp_model = model_cls()
    ddp_model.training_step = ddp_model.training_step_normal
    ddp_model.validation_step = ddp_model.validation_step_normal
    ddp_model.train_dataloader = ddp_model.train_dataloader_normal
    ddp_model.val_dataloader = ddp_model.val_dataloader_normal
    ddp_model_clone = deepcopy(ddp_model)

    trainer = Trainer(
        fast_dev_run=True,
        max_epochs=1,
        gpus=gpus,
        precision=precision,
        accelerator=accelerator,
    )

    max_ddp_memory, ddp_time = record_ddp_fit_model_stats(
        trainer=trainer,
        model=ddp_model,
        gpus=gpus
    )

    for ddp_param, shard_param in zip(ddp_model_clone.parameters(), ddp_model.parameters()):
        assert not torch.equal(ddp_param, shard_param), (ddp_param.shape, shard_param.shape)

    # move previous model to cpu
    ddp_model = ddp_model.cpu()
    ddp_model_clone = ddp_model_clone.cpu()

    torch.cuda.empty_cache()
    torch_distrib.barrier()

    # Reset and train pipe DDP batch_size 2 to emulate previous DDP with batch_size 1.
    seed_everything(42)
    pipe_model = model_cls()
    pipe_model.training_step = pipe_model.training_step_pipe
    pipe_model.validation_step = pipe_model.validation_step_pipe
    pipe_model.train_dataloader = pipe_model.train_dataloader_pipe
    pipe_model.val_dataloader = pipe_model.val_dataloader_pipe

    pipe_model_clone = deepcopy(pipe_model)

    trainer = Trainer(
        max_epochs=1,
        gpus=gpus,
        precision=precision,
        accelerator=accelerator,
        plugins=[PipePlugin()],
        limit_train_batches=1,
        accumulate_grad_batches=1,
    )

    max_pipe_memory, pipe_time = record_ddp_fit_model_stats(
        trainer=trainer,
        model=pipe_model,
        gpus=gpus
    )

    print(torch_distrib.get_rank(), trainer.get_model())

    for ddp_param, shard_param in zip(pipe_model_clone.parameters(), pipe_model.parameters()):
        assert not torch.equal(ddp_param, shard_param), (ddp_param.shape, shard_param.shape)

    if torch_distrib.get_rank() == 0:
        for ddp_param, shard_param in zip(list(ddp_model.parameters()), pipe_model.parameters()):
            assert torch.equal(ddp_param, shard_param), (ddp_param.shape, shard_param.shape)

    # Assert model parameters are identical after fit
    if torch_distrib.get_rank() == 1:
        for ddp_param, shard_param in zip(list(ddp_model.parameters())[::-1], pipe_model.parameters()):
            assert torch.equal(ddp_param, shard_param), (ddp_param.shape, shard_param.shape)

    # Assert speed parity by ensuring percentage difference between pipe/ddp is below threshold
    percent_diff = (pipe_time - ddp_time_2) / pipe_time

    assert percent_diff <= max_percent_speed_diff, \
        f'pipe plugin was too slow compared to DDP, pipe Time: {pipe_time}, DDP Time: {ddp_time_2}'

    # Assert CUDA memory parity
    print(f"rank: {torch_distrib.get_rank()}, max_pipe_memory: {max_pipe_memory}, max_ddp_memory: {max_ddp_memory_2} ")
    # assert max_pipe_memory <= max_ddp_memory


class PipeBoringModelV2(LightningModule):

    count = 0

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(torch.nn.Linear(32, 32, bias=False), nn.ReLU(), nn.Linear(32, 2, bias=False))

    def forward(self, x):
        return self.layers(x)

    def loss(self, prediction):
        # An arbitrary loss to have a loss that updates the model weights during `Trainer.fit` calls
        return torch.nn.functional.mse_loss(prediction, torch.ones_like(prediction))

    def training_step(self, batch, batch_idx):
        # manual
        opt = self.optimizers()
        output = self.layers(batch)
        print(output)
        loss = self.loss(output)
        self.manual_backward(loss, opt)
        self.manual_optimizer_step(opt)
        self.count += 1

    def validation_step(self, batch, batch_idx):
        output = self.layers(batch)
        loss = self.loss(output)

    def training_step_end(self, training_step_outputs):
        return training_step_outputs

    def train_dataloader(self):
        seed_everything(42)
        return torch.utils.data.DataLoader(RandomDataset(32, 1), batch_size=1)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(RandomDataset(32, 1), batch_size=1)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.layers.parameters(), lr=0.1)
        return optimizer

    @property
    def automatic_optimization(self) -> bool:
        return False


def run_pipe_correctness_version_2(
        accelerator='ddp_spawn',
        gpus=0,
        precision=32,
        max_percent_speed_diff=0.25,
        model_cls=PipeBoringModelV2):
    """
    Ensures that the trained model is identical to the standard DDP implementation.
    Also checks for speed/memory regressions, we should expect always less memory but performance to fluctuate.
    Args:
        accelerator: Accelerator type for test.
        gpus: Number of GPUS to enable.
        precision: Whether to use AMP or normal FP32 training.
        max_percent_speed_diff: The maximum speed difference compared to normal DDP training.
        This is more a safety net for variability in CI which can vary in speed, not for benchmarking.
        model_cls: Model class to use for test.
    """

    # Train normal DDP
    seed_everything(42)
    ddp_model = model_cls()
    ddp_model_clone = deepcopy(ddp_model)

    trainer = Trainer(
        fast_dev_run=True,
        max_epochs=1,
        gpus=gpus,
        precision=precision,
        accelerator=accelerator,
    )

    max_ddp_memory, ddp_time = record_ddp_fit_model_stats(
        trainer=trainer,
        model=ddp_model,
        gpus=gpus
    )

    for ddp_param, shard_param in zip(ddp_model_clone.parameters(), ddp_model.parameters()):
        assert not torch.equal(ddp_param, shard_param), (ddp_param.shape, shard_param.shape)

    torch.cuda.empty_cache()
    torch_distrib.barrier()

    # Reset and train pipe DDP
    seed_everything(42)
    pipe_model = model_cls()

    pipe_model_clone = deepcopy(pipe_model)

    trainer = Trainer(
        max_epochs=1,
        gpus=gpus,
        precision=precision,
        accelerator=accelerator,
        plugins=[PipePlugin(balance=[2, 1], version=2)],
    )

    max_pipe_memory, pipe_time = record_ddp_fit_model_stats(
        trainer=trainer,
        model=pipe_model,
        gpus=gpus
    )

    for ddp_param, shard_param in zip(pipe_model_clone.parameters(), pipe_model.parameters()):
        assert not torch.equal(ddp_param, shard_param), (ddp_param.shape, shard_param.shape)

    # Assert model parameters are identical after fit
    for ddp_param, shard_param in zip(list(ddp_model.parameters())[::-1], pipe_model.parameters()):
        assert torch.equal(ddp_param, shard_param), (ddp_param.shape, shard_param.shape)

        # Assert speed parity by ensuring percentage difference between pipe/ddp is below threshold
    percent_diff = (pipe_time - ddp_time) / pipe_time

    assert percent_diff <= max_percent_speed_diff, \
        f'pipe plugin was too slow compared to DDP, pipe Time: {pipe_time}, DDP Time: {ddp_time}'

    if gpus > 0:
        # Assert CUDA memory parity
        relative_memory = np.abs(max_pipe_memory - max_ddp_memory) / max_ddp_memory
        assert np.abs(max_pipe_memory - max_ddp_memory) / max_pipe_memory <= 0.02, \
            f'pipe plugin used too much memory compared to DDP,' \
            f'pipe Mem: {max_pipe_memory}, DDP Mem: {max_ddp_memory}'

    print("Version 2")
