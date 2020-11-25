import glob
import os
import platform
import time
from distutils.version import LooseVersion
from unittest import mock

import pytest
import torch
from torch.utils.data.distributed import DistributedSampler

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.plugins.sharded_native_amp_plugin import ShardedNativeAMPPlugin
from pytorch_lightning.plugins.sharded_plugin import DDPShardedPlugin, FAIRSCALE_AVAILABLE
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
@pytest.mark.skipif(not FAIRSCALE_AVAILABLE, reason="Fairscale is not available")
def test_ddp_choice_sharded(tmpdir, ddp_backend, gpus, num_processes):
    """
        Test to ensure that plugin is correctly chosen
    """

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
@pytest.mark.skipif(not FAIRSCALE_AVAILABLE, reason="Fairscale is not available")
def test_ddp_choice_sharded_amp(tmpdir, ddp_backend, gpus, num_processes):
    """
        Test to ensure that plugin native amp plugin is correctly chosen when using sharded
    """

    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert isinstance(trainer.accelerator_backend.ddp_plugin, DDPShardedPlugin)
            assert isinstance(trainer.precision_connector.backend, ShardedNativeAMPPlugin)
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(
        fast_dev_run=True,
        gpus=gpus,
        precision=16,
        num_processes=num_processes,
        distributed_backend=ddp_backend,
        plugins=[DDPShardedPlugin()],
        callbacks=[CB()],
    )

    with pytest.raises(SystemExit):
        trainer.fit(model)


@pytest.mark.skipif(platform.system() == "Windows",
                    reason="Distributed training is not supported on Windows")
@pytest.mark.skipif(not FAIRSCALE_AVAILABLE, reason="Fairscale is not available")
def test_ddp_sharded_plugin_checkpoint_cpu(tmpdir):
    """
        Test to ensure that checkpoint is saved correctly
    """
    model = BoringModel()
    trainer = Trainer(
        callbacks=[ModelCheckpoint(dirpath=tmpdir, save_last=True)],
        accelerator='ddp_cpu',
        plugins=[DDPShardedPlugin()],
        fast_dev_run=True,
    )

    trainer.fit(model)

    checkpoint_path = glob.glob(os.path.join(tmpdir, "*.ckpt"))[0]

    saved_model = BoringModel.load_from_checkpoint(checkpoint_path)

    # Assert model parameters are identical after loading
    for ddp_param, shard_param in zip(model.parameters(), saved_model.parameters()):
        assert torch.equal(ddp_param, shard_param)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.skipif(platform.system() == "Windows",
                    reason="Distributed training is not supported on Windows")
@pytest.mark.skipif(not FAIRSCALE_AVAILABLE, reason="Fairscale is not available")
def test_ddp_sharded_plugin_checkpoint_multi_gpu(tmpdir):
    """
        Test to ensure that checkpoint is saved correctly when using multiple GPUs
    """
    model = BoringModel()
    trainer = Trainer(
        callbacks=[ModelCheckpoint(dirpath=tmpdir, save_last=True)],
        gpus=2,
        accelerator='ddp_spawn',
        plugins=[DDPShardedPlugin()],
        fast_dev_run=True,
    )

    trainer.fit(model)

    checkpoint_path = glob.glob(os.path.join(tmpdir, "*.ckpt"))[0]

    saved_model = BoringModel.load_from_checkpoint(checkpoint_path)

    # Assert model parameters are identical after loading
    for ddp_param, shard_param in zip(model.parameters(), saved_model.parameters()):
        assert torch.equal(ddp_param, shard_param)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.skipif(platform.system() == "Windows",
                    reason="Distributed training is not supported on Windows")
@pytest.mark.skipif(not FAIRSCALE_AVAILABLE, reason="Fairscale is not available")
def test_ddp_sharded_plugin_finetune(tmpdir):
    """
        Test to ensure that we can save and restart training (simulate fine-tuning)
    """
    model = BoringModel()
    trainer = Trainer(
        callbacks=[ModelCheckpoint(dirpath=tmpdir, save_last=True)],
        gpus=2,
        accelerator='ddp_spawn',
        plugins=[DDPShardedPlugin()],
        fast_dev_run=True,
    )
    trainer.fit(model)

    checkpoint_path = glob.glob(os.path.join(tmpdir, "*.ckpt"))[0]
    saved_model = BoringModel.load_from_checkpoint(checkpoint_path)

    trainer = Trainer(
        callbacks=[ModelCheckpoint(dirpath=tmpdir, save_last=True)],
        fast_dev_run=True,
    )
    trainer.fit(saved_model)
    return 1


@pytest.mark.skipif(platform.system() == "Windows",
                    reason="Distributed training is not supported on Windows")
@pytest.mark.skipif(not FAIRSCALE_AVAILABLE, reason="Fairscale is not available")
def test_ddp_sharded_plugin_resume_from_checkpoint(tmpdir):
    """
        Test to ensure that resuming from checkpoint works
    """
    model = BoringModel()
    trainer = Trainer(
        callbacks=[ModelCheckpoint(dirpath=tmpdir, save_last=True)],
        accelerator='ddp_cpu',
        plugins=[DDPShardedPlugin()],
        fast_dev_run=True,
    )

    trainer.fit(model)

    checkpoint_path = glob.glob(os.path.join(tmpdir, "*.ckpt"))[0]

    model = BoringModel()

    trainer = Trainer(
        callbacks=[ModelCheckpoint(dirpath=tmpdir, save_last=True)],
        accelerator='ddp_cpu',
        plugins=[DDPShardedPlugin()],
        fast_dev_run=True,
        resume_from_checkpoint=checkpoint_path
    )

    trainer.fit(model)
    return 1


@pytest.mark.skip(reason="Currently unsupported restarting training on different number of devices.")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.skipif(platform.system() == "Windows",
                    reason="Distributed training is not supported on Windows")
@pytest.mark.skipif(not FAIRSCALE_AVAILABLE, reason="Fairscale is not available")
def test_ddp_sharded_plugin_resume_from_checkpoint_downsize_gpus(tmpdir):
    """
        Test to ensure that resuming from checkpoint works when downsizing number of GPUS
    """
    model = BoringModel()
    trainer = Trainer(
        callbacks=[ModelCheckpoint(dirpath=tmpdir, save_last=True)],
        accelerator='ddp_spawn',
        plugins=[DDPShardedPlugin()],
        fast_dev_run=True,
        gpus=2,
    )

    trainer.fit(model)

    checkpoint_path = glob.glob(os.path.join(tmpdir, "*.ckpt"))[0]

    model = BoringModel()

    trainer = Trainer(
        callbacks=[ModelCheckpoint(dirpath=tmpdir, save_last=True)],
        accelerator='ddp_spawn',
        plugins=[DDPShardedPlugin()],
        fast_dev_run=True,
        gpus=1,
        resume_from_checkpoint=checkpoint_path
    )

    trainer.fit(model)
    return 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU machine")
@pytest.mark.skipif(platform.system() == "Windows",
                    reason="Distributed training is not supported on Windows")
@pytest.mark.skipif(not FAIRSCALE_AVAILABLE, reason="Fairscale is not available")
def test_ddp_sharded_plugin_resume_from_checkpoint_gpu_to_cpu(tmpdir):
    """
        Test to ensure that resuming from checkpoint works when going from GPUs- > CPU
    """
    model = BoringModel()
    trainer = Trainer(
        callbacks=[ModelCheckpoint(dirpath=tmpdir, save_last=True)],
        accelerator='ddp_spawn',
        plugins=[DDPShardedPlugin()],
        gpus=1,
        fast_dev_run=True
    )

    trainer.fit(model)

    checkpoint_path = glob.glob(os.path.join(tmpdir, "*.ckpt"))[0]

    model = BoringModel()

    trainer = Trainer(
        callbacks=[ModelCheckpoint(dirpath=tmpdir, save_last=True)],
        plugins=[DDPShardedPlugin()],
        accelerator='ddp_cpu',
        fast_dev_run=True,
        resume_from_checkpoint=checkpoint_path
    )

    trainer.fit(model)
    return 1


@pytest.mark.skipif(platform.system() == "Windows",
                    reason="Distributed training is not supported on Windows")
@pytest.mark.skipif(not FAIRSCALE_AVAILABLE, reason="Fairscale is not available")
def test_ddp_sharded_plugin_correctness_one_device():
    # Allow slightly slower speed due to one CPU machine doing rigorously memory saving calls
    run_sharded_correctness(accelerator='ddp_cpu')


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU machine")
@pytest.mark.skipif(platform.system() == "Windows",
                    reason="Distributed training is not supported on Windows")
@pytest.mark.skipif(not FAIRSCALE_AVAILABLE, reason="Fairscale is not available")
def test_ddp_sharded_plugin_correctness_one_gpu():
    run_sharded_correctness(gpus=1, accelerator='ddp_spawn')


@pytest.mark.skipif(
    LooseVersion(torch.__version__) < LooseVersion("1.6.0"),
    reason="Minimal PT version is set to 1.6")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU machine")
@pytest.mark.skipif(platform.system() == "Windows",
                    reason="Distributed training is not supported on Windows")
@pytest.mark.skipif(not FAIRSCALE_AVAILABLE, reason="Fairscale is not available")
def test_ddp_sharded_plugin_correctness_amp_one_gpu():
    run_sharded_correctness(gpus=1, precision=16, accelerator='ddp_spawn')


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.skipif(platform.system() == "Windows",
                    reason="Distributed training is not supported on Windows")
@pytest.mark.skipif(not FAIRSCALE_AVAILABLE,
                    reason="Fairscale is not available")
def test_ddp_sharded_plugin_correctness_multi_gpu():
    run_sharded_correctness(gpus=2, accelerator='ddp_spawn')


@pytest.mark.skipif(
    LooseVersion(torch.__version__) < LooseVersion("1.6.0"),
    reason="Minimal PT version is set to 1.6")
@pytest.mark.skipif(platform.system() == "Windows",
                    reason="Distributed training is not supported on Windows")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.skipif(not FAIRSCALE_AVAILABLE, reason="Fairscale is not available")
def test_ddp_sharded_plugin_correctness_amp_multi_gpu():
    run_sharded_correctness(gpus=2, precision=16, accelerator='ddp_spawn')


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.skipif(platform.system() == "Windows",
                    reason="Distributed training is not supported on Windows")
@pytest.mark.skipif(not FAIRSCALE_AVAILABLE,
                    reason="Fairscale is not available")
def test_ddp_sharded_plugin_correctness_multi_gpu_multi_optim():
    """
        Ensures same results using multiple optimizers across multiple GPUs
    """
    run_sharded_correctness(
        gpus=2,
        accelerator='ddp_spawn',
        model_cls=TestMultipleOptimizersModel,
        max_percent_speed_diff=0.3  # Increase speed diff since only 2 GPUs sharding 2 optimizers
    )


@pytest.mark.skip(reason="Currently DDP manual optimization is broken due to no reduce within training step.")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.skipif(platform.system() == "Windows",
                    reason="Distributed training is not supported on Windows")
@pytest.mark.skipif(not FAIRSCALE_AVAILABLE,
                    reason="Fairscale is not available")
@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
def test_ddp_sharded_plugin_correctness_multi_gpu_multi_optim_manual(tmpdir):
    """
        Ensures using multiple optimizers across multiple GPUs with manual optimization
    """
    run_sharded_correctness(
        gpus=2,
        accelerator='ddp_spawn',
        model_cls=TestManualModel,
    )


class TestModel(BoringModel):
    """
        Overrides training loader to ensure we enforce the same seed for all DDP processes.
    """

    def train_dataloader(self):
        seed_everything(42)
        return torch.utils.data.DataLoader(RandomDataset(32, 64))


class TestManualModel(TestModel):
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


class TestMultipleOptimizersModel(TestModel):
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
        model_cls=TestModel):
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
