import os
import platform
from unittest import mock

import pytest
import torch
from torch.utils.data.distributed import DistributedSampler

from benchmarks.utilities import plugin_parity_test
from pytorch_lightning import seed_everything
from pytorch_lightning.plugins.sharded_plugin import DDPShardedPlugin
from pytorch_lightning.utilities import FAIRSCALE_AVAILABLE, NATIVE_AMP_AVAILABLE
from tests.backends.launcher import DDPLauncher
from tests.base.boring_model import BoringModel, RandomDataset


@pytest.mark.skipif(platform.system() == "Windows",
                    reason="Distributed training is not supported on Windows")
@pytest.mark.skipif(not FAIRSCALE_AVAILABLE, reason="Fairscale is not available")
def test_ddp_sharded_plugin_correctness_one_device():
    # Allow slightly slower speed due to one CPU doing additional sequential memory saving calls
    plugin_parity_test(
        accelerator='ddp_cpu',
        max_percent_speed_diff=0.5,
        plugin=DDPShardedPlugin(),
        model_cls=SeedTrainLoaderModel
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU machine")
@pytest.mark.skipif(platform.system() == "Windows",
                    reason="Distributed training is not supported on Windows")
@pytest.mark.skipif(not FAIRSCALE_AVAILABLE, reason="Fairscale is not available")
def test_ddp_sharded_plugin_correctness_one_gpu():
    plugin_parity_test(
        gpus=1,
        accelerator='ddp_spawn',
        plugin=DDPShardedPlugin(),
        model_cls=SeedTrainLoaderModel
    )


@pytest.mark.skipif(not NATIVE_AMP_AVAILABLE, reason="Requires native AMP")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU machine")
@pytest.mark.skipif(platform.system() == "Windows",
                    reason="Distributed training is not supported on Windows")
@pytest.mark.skipif(not FAIRSCALE_AVAILABLE, reason="Fairscale is not available")
def test_ddp_sharded_plugin_correctness_amp_one_gpu():
    plugin_parity_test(
        gpus=1,
        precision=16,
        accelerator='ddp_spawn',
        plugin=DDPShardedPlugin(),
        model_cls=SeedTrainLoaderModel
    )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.skipif(platform.system() == "Windows",
                    reason="Distributed training is not supported on Windows")
@pytest.mark.skipif(not FAIRSCALE_AVAILABLE, reason="Fairscale is not available")
def test_ddp_sharded_plugin_correctness_multi_gpu():
    plugin_parity_test(
        gpus=2,
        accelerator='ddp_spawn',
        plugin=DDPShardedPlugin(),
        model_cls=SeedTrainLoaderModel
    )


@pytest.mark.skipif(not NATIVE_AMP_AVAILABLE, reason="Requires native AMP")
@pytest.mark.skipif(platform.system() == "Windows",
                    reason="Distributed training is not supported on Windows")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.skipif(not FAIRSCALE_AVAILABLE, reason="Fairscale is not available")
def test_ddp_sharded_plugin_correctness_amp_multi_gpu():
    plugin_parity_test(
        gpus=2,
        precision=16,
        accelerator='ddp_spawn',
        plugin=DDPShardedPlugin(),
        model_cls=SeedTrainLoaderModel
    )


@pytest.mark.skipif(not FAIRSCALE_AVAILABLE, reason="Fairscale is not available")
@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.skipif(not os.getenv("PL_RUNNING_SPECIAL_TESTS", '0') == '1',
                    reason="test should be run outside of pytest")
@DDPLauncher.run("--distributed_backend ddp --gpus 2 --precision 32")
def test_ddp_sharded_plugin_correctness_multi_gpu_ddp(tmpdir, args=None):
    plugin_parity_test(
        gpus=args.gpus,
        precision=args.precision,
        accelerator=args.distributed_backend,
        plugin=DDPShardedPlugin(),
        model_cls=SeedTrainLoaderModel
    )


@pytest.mark.skipif(not FAIRSCALE_AVAILABLE, reason="Fairscale is not available")
@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.skipif(not os.getenv("PL_RUNNING_SPECIAL_TESTS", '0') == '1',
                    reason="test should be run outside of pytest")
@DDPLauncher.run("--distributed_backend ddp --gpus 2  --precision 16")
def test_ddp_sharded_plugin_correctness_amp_multi_gpu_ddp(tmpdir, args=None):
    plugin_parity_test(
        gpus=args.gpus,
        precision=args.precision,
        accelerator=args.distributed_backend,
        plugin=DDPShardedPlugin(),
        model_cls=SeedTrainLoaderModel
    )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.skipif(platform.system() == "Windows",
                    reason="Distributed training is not supported on Windows")
@pytest.mark.skipif(not FAIRSCALE_AVAILABLE, reason="Fairscale is not available")
def test_ddp_sharded_plugin_correctness_multi_gpu_multi_optim():
    """
        Ensures same results using multiple optimizers across multiple GPUs
    """
    plugin_parity_test(
        plugin=DDPShardedPlugin(),
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
    plugin_parity_test(
        plugin=DDPShardedPlugin(),
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
