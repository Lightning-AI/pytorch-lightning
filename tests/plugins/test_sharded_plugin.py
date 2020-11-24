import os
import platform
import time
from distutils.version import LooseVersion
from unittest import mock

import pytest
import torch
from torch.utils.data.distributed import DistributedSampler

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.plugins.sharded_plugin import DDPShardedPlugin
from tests.base.boring_model import BoringModel, RandomDataset


@mock.patch.dict(
    os.environ,
    {
        "CUDA_VISIBLE_DEVICES": "0,1",
        "SLURM_NTASKS": "2",
        "SLURM_JOB_NAME": "SOME_NAME",
        "SLURM_NODEID": "0",
        "LOCAL_RANK": "0",
        "SLURM_LOCALID": "0",
    },
)
@mock.patch("torch.cuda.device_count", return_value=2)
@pytest.mark.parametrize(
    ["ddp_backend", "gpus", "num_processes"],
    [("ddp_cpu", None, None), ("ddp", 2, 0), ("ddp2", 2, 0), ("ddp_spawn", 2, 0)],
)
def test_ddp_choice_sharded_cpu(tmpdir, ddp_backend, gpus, num_processes):
    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert isinstance(trainer.accelerator_backend.ddp_plugin, DDPShardedPlugin)
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(
        fast_dev_run=True,
        gpus=gpus,
        num_processes=num_processes,
        distributed_backend=ddp_backend,
        plugins=[DDPShardedPlugin()],
        callbacks=[CB()],
    )

    with pytest.raises(SystemExit):
        trainer.fit(model)


@pytest.mark.skipif(platform.system() == "Windows",
                    reason="Distributed training is not supported on Windows")
def test_ddp_sharded_plugin_correctness_one_device():
    run_sharded_correctness(accelerator='ddp_cpu')


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU machine")
@pytest.mark.skipif(platform.system() == "Windows",
                    reason="Distributed training is not supported on Windows")
def test_ddp_sharded_plugin_correctness_one_gpu():
    run_sharded_correctness(gpus=1, accelerator='ddp_spawn')


@pytest.mark.skipif(
    LooseVersion(torch.__version__) < LooseVersion("1.6.0"),
    reason="Minimal PT version is set to 1.6")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU machine")
@pytest.mark.skipif(platform.system() == "Windows",
                    reason="Distributed training is not supported on Windows")
def test_ddp_sharded_plugin_correctness_amp_one_gpu():
    run_sharded_correctness(gpus=1, precision=16, accelerator='ddp_spawn')


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.skipif(platform.system() == "Windows",
                    reason="Distributed training is not supported on Windows")
def test_ddp_sharded_plugin_correctness_multi_gpu():
    run_sharded_correctness(gpus=2, accelerator='ddp_spawn')


@pytest.mark.skipif(
    LooseVersion(torch.__version__) < LooseVersion("1.6.0"),
    reason="Minimal PT version is set to 1.6")
@pytest.mark.skipif(platform.system() == "Windows",
                    reason="Distributed training is not supported on Windows")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_ddp_sharded_plugin_correctness_amp_multi_gpu():
    run_sharded_correctness(gpus=2, precision=16, accelerator='ddp_spawn')


class TestModel(BoringModel):
    """
        Overrides training loader to ensure we enforce the same seed for all DDP processes.
    """

    def train_dataloader(self):
        seed_everything(42)
        return torch.utils.data.DataLoader(RandomDataset(32, 64))


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
    if gpus > 0:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    time_start = time.perf_counter()
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
        max_percent_speed_regression=0.1):
    """
        Ensures that the trained model is identical to the standard DDP implementation.
        Also checks for speed/memory regressions, we should expect always less memory but performance to fluctuate.
    Args:
        accelerator: Accelerator type for test.
        gpus: Number of GPUS to enable.
        precision: Whether to use AMP or normal FP32 training.
        max_percent_speed_regression: The maximum speed regression compared to normal DDP training

    """

    # Train normal DDP
    seed_everything(42)
    ddp_model = TestModel()

    trainer = Trainer(
        limit_val_batches=0.0,
        fast_dev_run=False,
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
    sharded_model = TestModel()

    trainer = Trainer(
        limit_val_batches=0.0,
        fast_dev_run=False,
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
        assert torch.equal(ddp_param, shard_param)

    # Assert speed parity
    upper_bound_speed = ddp_time * (1 + max_percent_speed_regression)
    assert sharded_time <= upper_bound_speed

    if gpus > 0:
        # Assert CUDA memory parity
        assert max_sharded_memory <= max_ddp_memory
