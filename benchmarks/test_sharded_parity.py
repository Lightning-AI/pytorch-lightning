import os
import platform
import time
from unittest import mock

import pytest
import torch
from torch.utils.data.distributed import DistributedSampler

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.plugins.sharded_plugin import DDPShardedPlugin
from pytorch_lightning.utilities import FAIRSCALE_AVAILABLE, NATIVE_AMP_AVALAIBLE
from tests.backends.launcher import DDPLauncher
from tests.base.boring_model import BoringModel, RandomDataset


@pytest.mark.skipif(platform.system() == "Windows",
                    reason="Distributed training is not supported on Windows")
@pytest.mark.skipif(not FAIRSCALE_AVAILABLE, reason="Fairscale is not available")
def test_ddp_sharded_plugin_correctness_one_device():
    # Allow slightly slower speed due to one CPU doing additional sequential memory saving calls
    run_sharded_correctness(accelerator='ddp_cpu', max_percent_speed_diff=0.5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU machine")
@pytest.mark.skipif(platform.system() == "Windows",
                    reason="Distributed training is not supported on Windows")
@pytest.mark.skipif(not FAIRSCALE_AVAILABLE, reason="Fairscale is not available")
def test_ddp_sharded_plugin_correctness_one_gpu():
    run_sharded_correctness(gpus=1, accelerator='ddp_spawn')


@pytest.mark.skipif(not NATIVE_AMP_AVALAIBLE, reason="Requires native AMP")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU machine")
@pytest.mark.skipif(platform.system() == "Windows",
                    reason="Distributed training is not supported on Windows")
@pytest.mark.skipif(not FAIRSCALE_AVAILABLE, reason="Fairscale is not available")
def test_ddp_sharded_plugin_correctness_amp_one_gpu():
    run_sharded_correctness(gpus=1, precision=16, accelerator='ddp_spawn')


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.skipif(platform.system() == "Windows",
                    reason="Distributed training is not supported on Windows")
@pytest.mark.skipif(not FAIRSCALE_AVAILABLE, reason="Fairscale is not available")
def test_ddp_sharded_plugin_correctness_multi_gpu():
    run_sharded_correctness(gpus=2, accelerator='ddp_spawn')


@pytest.mark.skipif(not NATIVE_AMP_AVALAIBLE, reason="Requires native AMP")
@pytest.mark.skipif(platform.system() == "Windows",
                    reason="Distributed training is not supported on Windows")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.skipif(not FAIRSCALE_AVAILABLE, reason="Fairscale is not available")
def test_ddp_sharded_plugin_correctness_amp_multi_gpu():
    run_sharded_correctness(gpus=2, precision=16, accelerator='ddp_spawn')


@pytest.mark.skipif(not FAIRSCALE_AVAILABLE, reason="Fairscale is not available")
@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.skipif(not os.getenv("PL_RUNNING_SPECIAL_TESTS", '0') == '1',
                    reason="test should be run outside of pytest")
@DDPLauncher.run("--distributed_backend ddp --gpus 2 --precision 32")
def test_ddp_sharded_plugin_correctness_multi_gpu_ddp(tmpdir, args=None):
    run_sharded_correctness(gpus=args.gpus, precision=args.precision, accelerator=args.distributed_backend)


@pytest.mark.skipif(not FAIRSCALE_AVAILABLE, reason="Fairscale is not available")
@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.skipif(not os.getenv("PL_RUNNING_SPECIAL_TESTS", '0') == '1',
                    reason="test should be run outside of pytest")
@DDPLauncher.run("--distributed_backend ddp --gpus 2  --precision 16")
def test_ddp_sharded_plugin_correctness_amp_multi_gpu_ddp(tmpdir, args=None):
    run_sharded_correctness(gpus=args.gpus, precision=args.precision, accelerator=args.distributed_backend)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.skipif(platform.system() == "Windows",
                    reason="Distributed training is not supported on Windows")
@pytest.mark.skipif(not FAIRSCALE_AVAILABLE, reason="Fairscale is not available")
def test_ddp_sharded_plugin_correctness_multi_gpu_multi_optim():
    """
        Ensures same results using multiple optimizers across multiple GPUs
    """
    run_sharded_correctness(
        gpus=2,
        accelerator='ddp_spawn',
        model_cls=SeedTrainLoaderMultipleOptimizersModel,
        max_percent_speed_diff=0.3  # Increase speed diff since only 2 GPUs sharding 2 optimizers
    )


@pytest.mark.skip(reason="Currently DDP manual optimization is broken due to no reduce within training step.")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.skipif(platform.system() == "Windows",
                    reason="Distributed training is not supported on Windows")
@pytest.mark.skipif(not FAIRSCALE_AVAILABLE, reason="Fairscale is not available")
@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
def test_ddp_sharded_plugin_correctness_multi_gpu_multi_optim_manual(tmpdir):
    """
        Ensures using multiple optimizers across multiple GPUs with manual optimization
    """
    run_sharded_correctness(
        gpus=2,
        accelerator='ddp_spawn',
        model_cls=SeedTrainLoaderManualModel,
    )


class SeedTrainLoaderModel(BoringModel):
    """
        Overrides training loader to ensure we enforce the same seed for all DDP processes.
    """

    def train_dataloader(self):
        seed_everything(42)
        return torch.utils.data.DataLoader(RandomDataset(32, 64))


class SeedTrainLoaderManualModel(SeedTrainLoaderModel):
    def training_step(self, batch, batch_idx, optimizer_idx):
        # manual
        (opt_a, opt_b) = self.optimizers()
        loss_1 = self.step(batch)

        self.manual_backward(loss_1, opt_a)
        self.manual_optimizer_step(opt_a)

        # fake discriminator
        loss_2 = self.step(batch[0])

        # ensure we forward the correct params to the optimizer
        # without retain_graph we can't do multiple backward passes
        self.manual_backward(loss_2, opt_b, retain_graph=True)
        self.manual_backward(loss_2, opt_a, retain_graph=True)
        self.manual_optimizer_step(opt_b)

        assert self.layer.weight.grad is None or torch.all(self.layer.weight.grad == 0)

    def training_epoch_end(self, outputs) -> None:
        # outputs should be an array with an entry per optimizer
        assert len(outputs) == 2

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
        optimizer_2 = torch.optim.SGD(self.layer.parameters(), lr=0.1)
        return optimizer, optimizer_2

    @property
    def automatic_optimization(self) -> bool:
        return False


class SeedTrainLoaderMultipleOptimizersModel(SeedTrainLoaderModel):
    def training_step(self, batch, batch_idx, optimizer_idx):
        output = self.layer(batch)
        loss = self.loss(batch, output)
        return {"loss": loss}

    def training_epoch_end(self, outputs) -> None:
        # outputs should be an array with an entry per optimizer
        assert len(outputs) == 2

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
        optimizer_2 = torch.optim.SGD(self.layer.parameters(), lr=0.1)
        return optimizer, optimizer_2


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


def run_sharded_correctness(
        accelerator='ddp_spawn',
        gpus=0,
        precision=32,
        max_percent_speed_diff=0.25,
        model_cls=SeedTrainLoaderModel):
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

    # Reset and train sharded DDP
    seed_everything(42)
    sharded_model = model_cls()

    trainer = Trainer(
        fast_dev_run=True,
        max_epochs=1,
        gpus=gpus,
        precision=precision,
        accelerator=accelerator,
        plugins=[DDPShardedPlugin()],
    )

    max_sharded_memory, sharded_time = record_ddp_fit_model_stats(
        trainer=trainer,
        model=sharded_model,
        gpus=gpus
    )

    # Assert model parameters are identical after fit
    for ddp_param, shard_param in zip(ddp_model.parameters(), sharded_model.parameters()):
        assert torch.equal(ddp_param, shard_param), 'Model parameters are different between DDP and Sharded plugin'

    # Assert speed parity by ensuring percentage difference between sharded/ddp is below threshold
    percent_diff = (sharded_time - ddp_time) / sharded_time

    assert percent_diff <= max_percent_speed_diff, \
        f'Sharded plugin was too slow compared to DDP, Sharded Time: {sharded_time}, DDP Time: {ddp_time}'

    if gpus > 0:
        # Assert CUDA memory parity
        assert max_sharded_memory <= max_ddp_memory, \
            f'Sharded plugin used too much memory compared to DDP,' \
            f'Sharded Mem: {max_sharded_memory}, DDP Mem: {max_ddp_memory}'
